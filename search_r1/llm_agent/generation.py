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
        )['input_ids']

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

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding    
        if next_obs_ids is not None:    
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ], pad_to_left=False)
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, :max_len],
            'position_ids': new_position_ids[:, :max_len],
            'attention_mask': new_attention_mask[:, :max_len]
        })
        new_rollings.batch = self.tensor_fn.cut_to_effective_len(
                new_rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids'],
                cut_left=False
            )
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

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

    def _build_rollings(self, gen_batch: DataProto, out_batch: DataProto) -> DataProto:
        """
        Build the rolling state for generation.
        """
        prompt_ids = gen_batch.batch['input_ids']          # [B, L0]
        gen_ids   = out_batch.batch['input_ids']           # [B, Lg]

        new_input_ids = self.tensor_fn.concatenate_with_padding([
                prompt_ids,
                gen_ids
            ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        max_len = self.config.max_prompt_length
        if new_input_ids.size(1) > max_len:
            new_input_ids      = new_input_ids[:, -max_len:]
            new_attention_mask = new_attention_mask[:, -max_len:]
            new_position_ids   = new_position_ids[:, -max_len:]

        outputs = DataProto.from_dict({
            'input_ids': new_input_ids,
            'attention_mask': new_attention_mask,
            'position_ids': new_position_ids
        })

        outputs.batch = self.tensor_fn.cut_to_effective_len(
                outputs.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
        return outputs

    # def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
    def run_llm_loop(self, gen_batch) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.ones(batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]

        max_start = self.config.max_start_length
        gen_batch.batch = {k: v[:, -max_start:] for k, v in gen_batch.batch.items()}
        out_batch = DataProto.from_dict({k: v[:, :0] for k, v in gen_batch.batch.items()})
        final_batch = DataProto.from_dict({k: v[:, :0] for k, v in gen_batch.batch.items()})

        gen_batch.batch = self.tensor_fn.cut_to_effective_len(
                gen_batch.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
        max_turns = self.config.max_turns
        # breakpoint()
        for step in range(max_turns + 1):
            if not active_mask.sum():
                break

            rollings = self._build_rollings(gen_batch, out_batch)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })       

            rollings_output = self._generate_with_gpu_padding(rollings_active)  ## ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs']
            responses_ids, responses_str = self._postprocess_responses(rollings_output.batch['responses'])  
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=step!=max_turns
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)

            out_batch = self._update_rolling_state(
                out_batch,
                responses_ids,
                next_obs_ids if step != max_turns else None
            )

        final_batch.meta_info = rollings_output.meta_info
        final_batch.batch["prompts"] = gen_batch.batch['input_ids']
        final_batch.batch["responses"] = out_batch.batch['input_ids']
        final_batch.batch["input_ids"] = torch.cat([
            gen_batch.batch['input_ids'],
            out_batch.batch['input_ids']
        ], dim=1)

        final_batch.batch["attention_mask"] = torch.cat([
            gen_batch.batch['attention_mask'],
            out_batch.batch['attention_mask']
        ], dim=1)

        final_batch.batch["position_ids"] = self.tensor_fn.create_position_ids(final_batch.batch["attention_mask"])
        final_batch.batch['info_mask'] = self._build_info_mask(
            gen_batch, 
            out_batch
        )

        final_batch.meta_info['turns_stats'] = turns_stats.tolist()
        final_batch.meta_info['active_mask'] = active_mask.tolist()
        final_batch.meta_info['valid_action_stats'] = valid_action_stats.tolist()
        final_batch.meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return final_batch

    def _build_info_mask(self, gen_batch: DataProto, out_batch: DataProto) -> torch.Tensor:
        info_mask = out_batch.batch['attention_mask'].clone()
        out_texts = self.tokenizer.batch_decode(
            out_batch.batch['input_ids'],
            skip_special_tokens=False
        )

        for i, text in enumerate(out_texts):
            for match in re.finditer(r"<information>.*?</information>", text, re.DOTALL):
                start_char, end_char = match.span()

                token_spans = self.tokenizer(
                    text[:end_char],
                    add_special_tokens=False,
                    return_offsets_mapping=True
                )["offset_mapping"]

                start_token_idx = next(
                    idx for idx, (s, e) in enumerate(token_spans) if s >= start_char
                )
                end_token_idx = max(
                    idx for idx, (s, e) in enumerate(token_spans) if e <= end_char
                )

                info_mask[i, start_token_idx:end_token_idx+1] = 0

        final_info_mask = torch.cat([
            gen_batch.batch['attention_mask'],
            info_mask
        ], dim=1)

        return final_info_mask

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
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
