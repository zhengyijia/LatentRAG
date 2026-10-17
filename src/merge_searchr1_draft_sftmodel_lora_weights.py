import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict

from model.searchr1_draft_sft_model import SearchR1DraftSFTModel

logger = logging.getLogger(__name__)

MODEL_CLASS_DICT = {
    'SearchR1DraftSFTModel': SearchR1DraftSFTModel,
}

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    # print initial config
    logger.info(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # load model
    model_cls = MODEL_CLASS_DICT.get(config.model.model_class, None)
    if model_cls is None:
        raise NotImplementedError(f"Model {config.model.model_class} is not implemented.")
    model = model_cls.load(config)
    
    model.base_model.save_pretrained(config.model.save_path)
    model.tokenizer.save_pretrained(config.model.save_path)

    logger.info(f"Merged model weights saved to {config.model.save_path}")

if __name__ == "__main__":
    main()


