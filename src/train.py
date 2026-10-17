import logging
import os
import gc
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from datetime import datetime
import torch
import torch.distributed as dist
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
import shutil
from transformers import TrainingArguments, TrainerCallback
import wandb
from model.latentrag import LatentRAG
from retriever.encoder import Encoder
from dataset import get_dataset
from collator import IdentityCollator
from trainer import CustomTrainer

logger = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    OmegaConf.resolve(config)
    
    # get slurm info
    slurm_info = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "job_name": os.getenv("SLURM_JOB_NAME"),
        "node_name": os.environ.get("SLURMD_NODENAME"),
        "node_list": os.getenv("SLURM_NODELIST"),
        "num_nodes": os.getenv("SLURM_JOB_NUM_NODES"),
        "ntasks": os.getenv("SLURM_NTASKS"),
        "local_nodeid": os.getenv("SLURM_NODEID"),
        "procid": os.getenv("SLURM_PROCID"),
        "local_procid": os.getenv("SLURM_LOCALID"),
    }
    with open_dict(config):
        config.slurm_info = slurm_info
        
    if getattr(config.custom_training, 'per_device_train_batch_size_scale', 1) != 1:
        logger.info(f"Scaling up per_device_train_batch_size by {config.custom_training.per_device_train_batch_size_scale}x")
        config.training.per_device_train_batch_size = int(config.training.per_device_train_batch_size * config.custom_training.per_device_train_batch_size_scale)
        logger.info(f"New per_device_train_batch_size: {config.training.per_device_train_batch_size}")
    
    # set up training args
    # if torch.cuda.device_count() <= 1:
    #     config.training.deepspeed = None  # disable deepspeed for single GPU
    training_args_dict = OmegaConf.to_container(config.training, resolve=True)
    del training_args_dict['overwrite_output_dir']
    training_args = TrainingArguments(**training_args_dict, 
                                      log_on_each_node=False,  # only log on main node
                                      )
    torch.cuda.set_device(training_args.local_rank)
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        # timestamp = 'DEBUG'
        timestamp = f"DEBUG_{timestamp}"
    
    # parameter tag for output_dir
    param_tag = f"_lr{training_args.learning_rate}"
    param_tag += f"_bs{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size}"
    param_tag += f"_ep{training_args.num_train_epochs}"
    param_tag += f"_lora_r{getattr(config.model.lora, 'lora_r', 0)}"
    param_tag += f"_dp{getattr(config.model.lora, 'lora_dropout', 0.0)}"
    param_tag += f"_{config.custom_training.align_loss}"
    param_tag += f"_temp{config.custom_training.temperature}"
    param_tag += f"_ls{getattr(config.custom_training, 'label_smoothing', 0.0)}"
    param_tag += f"_{config.model.projector_type}"
    
    if config.model.model_class in ("LatentRAG"): 
        encoder_trainable =  config.retriever.get("encoder_trainable", False)
        encoder_use_lora = config.retriever.get("encoder_lora", {}).get("use_lora", False)
        share_encoder = config.retriever.get("share_encoder", False)
        param_tag += f"_encoder_trainable" if encoder_trainable else ""
        param_tag += f"_encoder_lora" if encoder_use_lora else ""
        param_tag += f"_share_encoder" if share_encoder else ""
    
    if config.retriever.get("custom_encoder_name_or_path", None):
        param_tag += "_custom_encoder"
    
    num_latent_thought_tokens = config.model.get("num_latent_thought_tokens", 0)
    param_tag += f"_thought_tokens{num_latent_thought_tokens}"
    num_latent_subquery_tokens = config.model.get("num_latent_subquery_tokens", 0)
    param_tag += f"_subquery_tokens{num_latent_subquery_tokens}"
    
    run_name_suffix = getattr(config.training, 'run_name', '')
    param_tag += f"_{run_name_suffix}" if run_name_suffix else ""
    
    additional_param_tag = ''
    align_loss_weight = getattr(config.custom_training, 'align_loss_weight', 1.0)
    gen_loss_weight = getattr(config.custom_training, 'gen_loss_weight', 1.0)
    self_decode_thought_gen_loss_weight = getattr(config.custom_training, 'self_decode_thought_gen_loss_weight', 1.0)
    self_decode_subquery_gen_loss_weight = getattr(config.custom_training, 'self_decode_subquery_gen_loss_weight', 1.0)
    additional_param_tag += f"_alw{align_loss_weight}_glw{gen_loss_weight}_sdtw{self_decode_thought_gen_loss_weight}_sdsw{self_decode_subquery_gen_loss_weight}"
    
    mse_loss_weight = getattr(config.custom_training, 'align_mse_loss_weight', 1.0)
    nce_loss_weight = getattr(config.custom_training, 'align_nce_loss_weight', 1.0)
    additional_param_tag += f"_mlw{mse_loss_weight}_nlw{nce_loss_weight}"
    
    load_lora = getattr(config.model.lora, 'lora_name_or_path', None)
    additional_param_tag += f"_load_lora" if load_lora is not None else ""
    
    load_projector = getattr(config.model, 'projector_path', None)
    additional_param_tag += f"_load_projector" if load_projector is not None else ""
    
    decode_subquery = getattr(config.custom_training, 'decode_subquery', False)
    additional_param_tag += f"_decode_subquery" if decode_subquery else ""
    
    decode_thought = getattr(config.custom_training, 'decode_thought', False)
    additional_param_tag += f"_decode_thought" if decode_thought else ""
    
    llm_only_query_embedding = getattr(config.model, 'llm_only_query_embedding', False)
    additional_param_tag += f"_llm_only_query_embedding" if llm_only_query_embedding else ""
    
    training_args.output_dir = os.path.join(
        training_args.output_dir, 
        f"{config.model.model_tag}_{config.retriever.retriever_tag}_{config.data.data_tag}", 
        f"{timestamp}{param_tag}", 
        f"{timestamp}{additional_param_tag}",
    )
    
    training_args.run_name = (
        f"{timestamp}_train"
        f"_{config.model.model_tag}_{config.retriever.retriever_tag}_{config.data.data_tag}"
        f"{'_' + training_args.run_name if training_args.run_name else ''}" # suffix
    )
        
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and not config.training.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to clear the existing directory."
        )
        
    # Setup logging
    logging.basicConfig(level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    # for debug
    if os.getenv("DEBUG_MODE", "false").lower() == "true":
        logger.info(">>>>>>>>>> USING DEBUG MODE <<<<<<<<<<")
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # print initial config
    logger.info(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    
    set_seed(training_args.seed)

    if config.model.model_class == "LatentRAG":
        model_cls = LatentRAG
    else:
        raise ValueError(f"Unknown model_class: {config.model.model_class}")

    model = model_cls.build(config)
    logger.info(f"Model dtype: {next(model.parameters()).dtype}, device: {next(model.parameters()).device}")
    tokenizer = model.get_tokenizer()

    if config.model.model_class in ("LatentRAG"): 
        encoder_trainable =  config.retriever.get("encoder_trainable", False)
        encoder_use_lora = config.retriever.get("encoder_lora", {}).get("use_lora", False)
        encoder = Encoder.build(config, trainable=encoder_trainable, use_lora=encoder_use_lora, custom_encoder=True)
        model.set_encoder(encoder)
    
    if config.model.model_class in ("LatentRAG") and config.retriever.get("share_encoder", False):
        reference_encoder = encoder
    else:
        reference_encoder = Encoder.build(config, trainable=False, use_lora=False)
    model.set_reference_encoder(reference_encoder)  # use the same encoder as reference

    # prepare datasets
    filter_em_correct = getattr(config.data, 'filter_em_correct', True)
    filter_cover_em_correct = getattr(config.data, 'filter_cover_em_correct', False)
    filter_llm_judge_correct = getattr(config.data, 'filter_llm_judge_correct', False)
    train_dataset = get_dataset(config.data.train_files, config, tokenizer, 
                                batch_size=training_args.per_device_train_batch_size, 
                                eval_batch_size=training_args.per_device_eval_batch_size,
                                shuffle=True, filter_em_correct=filter_em_correct, 
                                filter_cover_em_correct=filter_cover_em_correct, 
                                filter_llm_judge_correct=filter_llm_judge_correct,
                                split_train_eval_set=config.data.get("split_train_eval_set", False), 
                                num_eval_samples=config.data.get("num_eval_samples", 200), 
                                num_train_samples=config.data.get("num_train_samples", -1), 
                                )
    training_args.per_device_train_batch_size = 1  # handle batch_size in dataset
    
    eval_dataset = None
    if config.data.get("split_train_eval_set", False) and config.data.get("eval_files", []):
        raise ValueError("`split_train_eval_set` should not be set when `eval_files` is provided.")
    
    if config.data.get("split_train_eval_set", False):
        train_dataset, eval_dataset = train_dataset

    if getattr(config.data, 'eval_files', []):
        eval_dataset = get_dataset(config.data.eval_files, config, tokenizer, 
                                    batch_size=training_args.per_device_eval_batch_size, 
                                    shuffle=False, filter_em_correct=filter_em_correct, 
                                    filter_cover_em_correct=filter_cover_em_correct, 
                                    filter_llm_judge_correct=filter_llm_judge_correct,)
    training_args.per_device_eval_batch_size = 1  # handle batch_size in dataset

    collator = IdentityCollator()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=collator,
        processing_class=tokenizer, 
        experiment_config=config, 
    )

    if config.training.overwrite_output_dir and training_args.local_rank == 0:
        if os.path.exists(training_args.output_dir):
            logger.info(f"Cleaning output directory: {training_args.output_dir}")
            shutil.rmtree(training_args.output_dir)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    
    # # monkey patch torch.load for resume_from_checkpoint
    # original_torch_load = torch.load
    # def safe_load(*args, **kwargs):
    #     kwargs["weights_only"] = False
    #     return original_torch_load(*args, **kwargs)
    # torch.load = safe_load
    
    # add config to wandb, and add wandb_url to config
    class WandbConfigUpdateCallback(TrainerCallback):
        def __init__(self, config):
            self.config = config

        def on_train_begin(self, args, state, control, **kwargs):
            if wandb.run is not None:
                # record wandb_url for tracking
                wandb_url = wandb.run.url
                with open_dict(self.config):
                    self.config.wandb_url = wandb_url
                # update config to wandb
                wandb.config.update(
                    {'latentrag_config': OmegaConf.to_container(self.config, resolve=True)},
                    allow_val_change=True
                )
    trainer.add_callback(WandbConfigUpdateCallback(config))
    
    # trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.train()
    
    # if wandb.run is not None:
    #     # record wandb_url for tracking
    #     wandb_url = wandb.run.url       
    #     with open_dict(config):
    #         config.wandb_url = wandb_url
        
    #     wandb.config.update(
    #         {'latentrag_config': OmegaConf.to_container(config, resolve=True)}, 
    #         allow_val_change=True
    #     )
    
    # save the final model
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        OmegaConf.save(config=config, f=f"{training_args.output_dir}/training_config.yaml")

if __name__ == "__main__":
    main()
