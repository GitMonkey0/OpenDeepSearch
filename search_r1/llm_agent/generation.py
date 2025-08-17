import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from verl.utils.reward_score.qa_em_format import extract_solution, em_check

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids'].long()

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids.long()

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _cut_to_effective_len(self, input_ids: torch.Tensor, cut_off="left") -> torch.Tensor:
        """
        Cut input_ids to the effective length based on the attention mask.
        """
        attention_mask = self.tensor_fn.create_attention_mask(input_ids)
        effective_len = attention_mask.sum(dim=1).max()
        if cut_off == "left":
            return input_ids[:, -effective_len:]
        elif cut_off == "right":
            return input_ids[:, :effective_len]
        else:
            raise ValueError(f"Invalid cut_off value: {cut_off}. Use 'left' or 'right'.")
    
    def run_llm_loop(self, gen_batch, ground_truth=None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
 
        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]

        init_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:]
        init_input_ids = self._cut_to_effective_len(init_input_ids, cut_off="left")
        output_ids_list = []

        cur_input_ids = init_input_ids.clone()
        max_turns = self.config.max_turns if ground_truth is not None else self.config.max_turns * 2
        for step in range(max_turns + 1):
            if not active_mask.sum():
                break
            cur_input_ids = cur_input_ids[:, -self.config.max_prompt_length:]
            cur_input_ids = self._cut_to_effective_len(cur_input_ids, cut_off="left")

            inputs = DataProto.from_dict({
                'input_ids': cur_input_ids,
                'attention_mask': self.tensor_fn.create_attention_mask(cur_input_ids),
                'position_ids': self.tensor_fn.create_position_ids(self.tensor_fn.create_attention_mask(cur_input_ids))
            })

            inputs = DataProto.from_dict({
                k: v[active_mask] for k, v in inputs.batch.items()
            })       

            outputs = self._generate_with_gpu_padding(inputs)

            responses_ids, responses_str = self._postprocess_responses(outputs.batch['responses'])  
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            responses_ids = self._cut_to_effective_len(responses_ids, cut_off="right")

            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, active_mask, do_search=step!=max_turns
            )
            next_obs_ids = self._process_next_obs(next_obs)
            next_obs_ids = self._cut_to_effective_len(next_obs_ids, cut_off="right")

            cur_output_ids = self.tensor_fn.concatenate_with_padding([
                responses_ids,
                next_obs_ids
            ], pad_to_left=False) if step!=self.config.max_turns else responses_ids
            cur_output_ids = self._cut_to_effective_len(cur_output_ids, cut_off="right")
            output_ids_list.append(cur_output_ids)

            cur_input_ids = self.tensor_fn.concatenate_with_padding([
                cur_input_ids,
                cur_output_ids
            ])
            cur_input_ids = self._cut_to_effective_len(cur_input_ids, cut_off="left")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

        output_ids = self.tensor_fn.concatenate_with_padding(output_ids_list, pad_to_left=False)
        output_ids = self._cut_to_effective_len(output_ids, cut_off="right")

        cur_input_ids = self.tensor_fn.concatenate_with_padding([
            init_input_ids,
            output_ids
        ])
        #######################################################################
        ## Reflection
        #######################################################################
        if ground_truth is not None:
            reflect_mask = self._create_reflect_mask(cur_input_ids, ground_truth)
            output_ids = self._create_reflect(output_ids, reflect_mask)

            cur_input_ids = self.tensor_fn.concatenate_with_padding([
                init_input_ids,
                output_ids
            ])
            reflect_output_ids_list = []
            for step in range(self.config.max_turns + 1):
                if not reflect_mask.sum():
                    break
                cur_input_ids = cur_input_ids[:, -self.config.max_prompt_length:]
                cur_input_ids = self._cut_to_effective_len(cur_input_ids, cut_off="left")

                inputs = DataProto.from_dict({
                    'input_ids': cur_input_ids,
                    'attention_mask': self.tensor_fn.create_attention_mask(cur_input_ids),
                    'position_ids': self.tensor_fn.create_position_ids(self.tensor_fn.create_attention_mask(cur_input_ids))
                })

                inputs = DataProto.from_dict({
                    k: v[reflect_mask] for k, v in inputs.batch.items()
                })       

                outputs = self._generate_with_gpu_padding(inputs)

                responses_ids, responses_str = self._postprocess_responses(outputs.batch['responses'])  
                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, reflect_mask)
                responses_ids = self._cut_to_effective_len(responses_ids, cut_off="right")

                next_obs, dones, valid_action, is_search = self.execute_predictions(
                    responses_str, reflect_mask, do_search=step!=self.config.max_turns
                )
                next_obs_ids = self._process_next_obs(next_obs)
                next_obs_ids = self._cut_to_effective_len(next_obs_ids, cut_off="right")

                cur_output_ids = self.tensor_fn.concatenate_with_padding([
                    responses_ids,
                    next_obs_ids
                ], pad_to_left=False)  if step!=self.config.max_turns else responses_ids
                cur_output_ids = self._cut_to_effective_len(cur_output_ids, cut_off="right")
                reflect_output_ids_list.append(cur_output_ids)

                cur_input_ids = self.tensor_fn.concatenate_with_padding([
                    cur_input_ids,
                    cur_output_ids
                ])
                cur_input_ids = self._cut_to_effective_len(cur_input_ids, cut_off="left")

                curr_reflect_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                reflect_mask = reflect_mask * curr_reflect_mask

            reflect_output_ids = self.tensor_fn.concatenate_with_padding(output_ids_list, pad_to_left=False)
            reflect_output_ids = self._cut_to_effective_len(reflect_output_ids, cut_off="right")

            final_output_ids = self.tensor_fn.concatenate_with_padding([output_ids, reflect_output_ids], pad_to_left=False)
        else:
            final_output_ids = output_ids
            
        final_output_ids = self._cut_to_effective_len(final_output_ids, cut_off="right")

        final_batch = DataProto.from_dict({k: v[:, :0] for k, v in gen_batch.batch.items()})
        final_batch.meta_info = outputs.meta_info
        final_batch.batch["prompts"] = init_input_ids
        final_batch.batch["responses"] = final_output_ids
        final_batch.batch["input_ids"] = torch.cat([
            init_input_ids, final_output_ids
        ], dim=1)

        final_batch.batch["attention_mask"] = self.tensor_fn.create_attention_mask(final_batch.batch["input_ids"])

        final_batch.batch["position_ids"] = self.tensor_fn.create_position_ids(final_batch.batch["attention_mask"])
        final_batch.batch['info_mask'] = self._create_info_mask(final_batch)

        final_batch.meta_info['turns_stats'] = turns_stats.tolist()
        final_batch.meta_info['active_mask'] = active_mask.tolist()
        final_batch.meta_info['valid_action_stats'] = valid_action_stats.tolist()
        final_batch.meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return final_batch

    def _create_reflect_mask(self, input_ids: torch.Tensor, ground_truth: List[str]) -> torch.Tensor:
        """
        Create a mask for the reflect phase based on ground truth.
        """
        input_str = self.tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True
        )
        reflect_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        
        for i, _gt in enumerate(ground_truth):
            gt = list(_gt)
            if not gt:
                continue
            answer = extract_solution(input_str[i])
            if answer is None:
                reflect_mask[i] = True
                continue
            if em_check(answer, gt) == 0:
                reflect_mask[i] = True

        return reflect_mask
    
    def _create_reflect(self, output_ids: torch.Tensor, reflect_mask: torch.Tensor) -> torch.Tensor:
        """
        Create a token for the reflect phase.
        """
        reflect_str = '\n<reflect>\nMaybe I should think, search and answer again?\n</reflect>\n'
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        new_ids = []
        pad_id = self.tokenizer.pad_token_id
        for text, mask in zip(decoded, reflect_mask):
            if mask:
                text = re.sub(r'<answer>.*?</answer>\s*$', '', text, flags=re.DOTALL) + reflect_str
            new_ids.append(self.tokenizer.encode(text, add_special_tokens=False))

        max_len = max(len(ids) for ids in new_ids)
        out = torch.full((output_ids.size(0), max_len), pad_id,
                        dtype=torch.long)
        for i, ids in enumerate(new_ids):
            out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def _create_info_mask(self, final_batch: DataProto) -> torch.Tensor:
        info_mask = final_batch.batch['attention_mask'].clone()
        full_input_ids = final_batch.batch['input_ids']
        prompts = final_batch.batch['prompts']

        batch_size = full_input_ids.size(0)
        for i in range(batch_size):
            prompt_len = prompts[i].size(0)
            full_text = self.tokenizer.decode(full_input_ids[i], skip_special_tokens=False)

            generated_text = self.tokenizer.decode(full_input_ids[i][prompt_len:], skip_special_tokens=False)

            for match in re.finditer(r"<information>.*?</information>", generated_text, re.DOTALL):
                start_char = len(self.tokenizer.decode(full_input_ids[i][:prompt_len], skip_special_tokens=False)) + match.start()
                end_char = len(self.tokenizer.decode(full_input_ids[i][:prompt_len], skip_special_tokens=False)) + match.end()

                token_spans = self.tokenizer(
                    full_text[:end_char],
                    add_special_tokens=False,
                    return_offsets_mapping=True
                )["offset_mapping"]

                start_token_idx = next(
                    idx for idx, (s, e) in enumerate(token_spans) if s >= start_char
                )
                end_token_idx = max(
                    idx for idx, (s, e) in enumerate(token_spans) if e <= end_char
                )

                info_mask[i, start_token_idx:end_token_idx + 1] = 0

        return info_mask

    def execute_predictions(self, predictions: List[str], active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\n\n<information>My previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.</information>\n\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
