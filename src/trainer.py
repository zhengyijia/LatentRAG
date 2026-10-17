import os
from typing import Optional
import psutil
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

import torch
from transformers.trainer import Trainer, TrainerCallback, TRAINING_ARGS_NAME
from transformers.integrations import WandbCallback
from transformers.integrations.integration_utils import rewrite_logs
from transformers.trainer_pt_utils import distributed_concat
import torch.distributed as dist
from transformers.utils import (
    is_accelerate_available,
    is_apex_available,
)
if is_accelerate_available():
    from accelerate.utils import DistributedType
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
from safetensors.torch import save_file, load_file

from model.latentrag import LatentRAG, TrainingOutput

import logging
logger = logging.getLogger(__name__)

class ShuffleCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, train_dataloader, **kwargs):
        dataset = train_dataloader.dataset
        if dataset.shuffle and hasattr(dataset, 'shuffle_and_rebatch'):
            epoch = int(state.epoch) if state.epoch is not None else 0
            dataset.shuffle_and_rebatch(epoch)

        return control

class CustomTrainer(Trainer):
    def __init__(self, experiment_config=None, *args, **kwargs):
        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.experiment_config = experiment_config
        
        self._current_epoch = -1
        self.add_callback(ShuffleCallback())
        
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

        self.log_train_metrics = ['gen_loss', 
                                  'self_decode_thought_gen_loss', 'self_decode_subquery_gen_loss',
                                  'align_loss', 'align_mse_loss', 'align_unnorm_mse_loss',
                                  'align_top1_nce_loss', 'align_topk_nce_loss', 'align_max_pos_nce_loss', 'align_min_pos_nce_loss',
                                  'align_per_topk_nce_loss', 'align_kl_rank_loss',
                                  'self_decode_thought_gen_accuracy', 'self_decode_thought_gen_accuracy_without_spec_tokens',
                                  'self_decode_subquery_gen_accuracy', 'self_decode_subquery_gen_accuracy_without_spec_tokens',
                                  'align_top1_accuracy', 'align_topk_accuracy', 'align_max_pos_accuracy', 
                                  'align_cosine_similarity', 'align_top1_cosine_similarity', 
                                  'align_topk_cosine_similarity', 'align_max_pos_cosine_similarity',
                                  'gen_accuracy', 'gen_accuracy_without_spec_tokens', ]
        self.log_eval_metrics = ['eval_gen_loss', 
                                 'eval_self_decode_thought_gen_loss', 'eval_self_decode_subquery_gen_loss',
                                 'eval_align_loss', 'eval_align_mse_loss', 'eval_align_unnorm_mse_loss',
                                 'eval_align_top1_nce_loss', 'eval_align_topk_nce_loss', 'eval_align_max_pos_nce_loss', 'eval_align_min_pos_nce_loss',
                                 'eval_align_per_topk_nce_loss', 'eval_align_kl_rank_loss',
                                 'eval_self_decode_thought_gen_accuracy', 'eval_self_decode_thought_gen_accuracy_without_spec_tokens',
                                 'eval_self_decode_subquery_gen_accuracy', 'eval_self_decode_subquery_gen_accuracy_without_spec_tokens',
                                 'eval_align_top1_accuracy', 'eval_align_topk_accuracy', 'eval_align_max_pos_accuracy',
                                 'eval_align_cosine_similarity', 'eval_align_top1_cosine_similarity', 
                                 'eval_align_topk_cosine_similarity', 'eval_align_max_pos_cosine_similarity',
                                 'eval_gen_accuracy', 'eval_gen_accuracy_without_spec_tokens',]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if getattr(self.accelerator, "parallelism_config", None) is not None:
            if self.accelerator.should_save_model:
                self._save(output_dir)
        # If we drop to here, we're in 1D parallelism, so all ranks need to go to `save_pretrained`
        elif (tp_size := getattr(self.model, "_tp_size", 0)) is not None and tp_size > 1:
            self._save(output_dir)
        elif self.is_fsdp_enabled:
            if "FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed) if self.deepspeed is not None else None
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir, state_dict=state_dict, save_safetensors=self.args.save_safetensors)
        
        if hasattr(self.model, "encoder") and self.model.encoder is not None:
            encoder_prefix = 'encoder.'
            encoder_state_dict = None
            if state_dict is not None:
                encoder_state_dict = {k[len(encoder_prefix):]: v for k, v in state_dict.items() if k.startswith(encoder_prefix)}
            self.model.encoder.save(os.path.join(output_dir, "encoder"), 
                                    state_dict=encoder_state_dict, save_safetensors=self.args.save_safetensors)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
        # save experiment config
        if self.experiment_config is not None:
            if wandb.run is not None and self.experiment_config.get('wandb_url', None) is None:
                wandb_url = wandb.run.url       
                with open_dict(self.experiment_config):
                    self.experiment_config.wandb_url = wandb_url
            OmegaConf.save(config=self.experiment_config, f=f"{output_dir}/training_config.yaml")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # # # TODO: for debug, skip early steps to identify bugs at later steps
        # if os.getenv("DEBUG_MODE", "false").lower() == "true": 
        #     skip_steps = getattr(self.experiment_config, 'skip_steps_for_debug', 0)
        #     if self.state.global_step < skip_steps:
        #         if self.model.training:
        #             loss = torch.tensor(0.0, device=inputs[0]['input_batch']['input_ids'].device, requires_grad=True)
        #         else:
        #             loss = torch.tensor(0.0, device=inputs[0]['input_batch']['input_ids'].device)
        #         outputs = TrainingOutput(loss=loss)
        #         return (loss, outputs)

        llm_input_batches, retrieval_data = inputs
        outputs = model(llm_input_batches=llm_input_batches,
                        retrieval_data=retrieval_data)
        loss = outputs.loss  # loss
        
        return (loss, outputs)

    
    # Copied from below & added log_train_metrics
    # https://github.com/huggingface/transformers/blob/4243bb844da660b387b3c409487e549754f7acc3/src/transformers/trainer.py#L3715
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
            
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            for metric_name in self.log_train_metrics:
                if not hasattr(outputs, metric_name) or getattr(outputs, metric_name) is None:
                    continue
                outputs[metric_name] = outputs[metric_name].mean()

        del inputs
        torch.cuda.empty_cache()
                
        kwargs = {}
        
        # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
        # https://github.com/huggingface/transformers/pull/35808
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(loss, **kwargs)
            
        # # TODO: for debug
        # for n,p in model.named_parameters():
        #     if "lora_" in n and p.grad is not None:
        #         if p.grad.abs().sum() > 0:
        #             print(n)
              
        # # TODO: for debug
        # from deepspeed.utils import safe_get_full_grad      
        # for n, p in model.named_parameters():
        #     if "lora_" in n and p.requires_grad:
        #         g = safe_get_full_grad(p)
        #         if g.abs().sum() > 0:
        #             print(n)
            
        # for logging
        for metric_name in self.log_train_metrics:
            if not hasattr(outputs, metric_name) or getattr(outputs, metric_name) is None:
                continue
            
            if hasattr(self.state, metric_name) and hasattr(self.state, f'{metric_name}_recorded_steps'):
                current = getattr(self.state, metric_name)
                recorded_steps = getattr(self.state, f'{metric_name}_recorded_steps')
            else:
                current = torch.tensor(0.0, device=loss.device)
                recorded_steps = 0
            setattr(self.state, metric_name, current + outputs[metric_name].detach())
            setattr(self.state, f'{metric_name}_recorded_steps', recorded_steps + 1)

        return loss.detach()
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs)
                
        prefix = "eval_"
                
        if self.args.n_gpu > 1:
            loss = loss.mean()
            for metric_name in self.log_eval_metrics:
                if not hasattr(outputs, metric_name[len(prefix):]) or getattr(outputs, metric_name[len(prefix):]) is None:
                    continue
                outputs[metric_name[len(prefix):]] = outputs[metric_name[len(prefix):]].mean()
            
        del inputs
        torch.cuda.empty_cache()
        
        # for logging
        for metric_name in self.log_eval_metrics:
            if not hasattr(outputs, metric_name[len(prefix):]) or getattr(outputs, metric_name[len(prefix):]) is None:
                continue
            
            if hasattr(self.state, metric_name) and hasattr(self.state, f'{metric_name}_eval_recorded_steps'):
                current = getattr(self.state, metric_name)
                eval_recorded_steps = getattr(self.state, f'{metric_name}_eval_recorded_steps')
            else:
                current = torch.tensor(0.0, device=loss.device)
                eval_recorded_steps = 0
            setattr(self.state, metric_name, current + outputs[metric_name[len(prefix):]].detach())
            setattr(self.state, f'{metric_name}_eval_recorded_steps', eval_recorded_steps + 1)
        
        return (loss, None, None)
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        is_distributed = (
            (self.args.distributed_state is not None and self.args.distributed_state.distributed_type != "NO") or 
            (self.args.distributed_state is None and self.args.local_rank != -1)
        )
        
        # -------- Training loss logging --------
        # Gather & avg across gpus like for actual loss
        # https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/trainer.py#L2257
        
        for metric_name in self.log_train_metrics:
            if hasattr(self.state, metric_name) and hasattr(self.state, f'{metric_name}_recorded_steps'):
                metric_value = getattr(self.state, metric_name)
                recorded_steps = getattr(self.state, f'{metric_name}_recorded_steps')
                
                if recorded_steps == 0:
                    continue
                
                # Gather & avg across gpus
                if is_distributed:
                    metric_value = distributed_concat(metric_value)
                
                metric_value = metric_value.mean().item()

                logs[metric_name] = round(metric_value / recorded_steps, 4)

        # clear state
        for metric_name in self.log_train_metrics:
            if hasattr(self.state, metric_name):
                delattr(self.state, metric_name)
            if hasattr(self.state, f'{metric_name}_recorded_steps'):
                delattr(self.state, f'{metric_name}_recorded_steps')

        # -------- Evaluation loss logging --------
        for metric_name in self.log_eval_metrics:
            if hasattr(self.state, metric_name) and hasattr(self.state, f'{metric_name}_eval_recorded_steps'):
                metric_value = getattr(self.state, metric_name)
                eval_recorded_steps = getattr(self.state, f'{metric_name}_eval_recorded_steps')
                
                if eval_recorded_steps == 0:
                    continue
                
                if is_distributed:
                    metric_value = distributed_concat(metric_value)
                    
                metric_value = metric_value.mean().item()
                
                logs[metric_name] = round(metric_value / eval_recorded_steps, 4)
        
        # clear state
        for metric_name in self.log_eval_metrics:
            if hasattr(self.state, metric_name):
                delattr(self.state, metric_name)
            if hasattr(self.state, f'{metric_name}_eval_recorded_steps'):
                delattr(self.state, f'{metric_name}_eval_recorded_steps')

        # -------- Other logging --------
        if "global_step" not in logs:
            logs["global_step"] = self.state.global_step

        if self.is_world_process_zero():
            super(CustomTrainer, self).log(logs, start_time)
    

