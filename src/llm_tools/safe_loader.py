import os
import logging
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig
from vllm import LLM
from requests.exceptions import RequestException
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)

class AutoModelSafeLoader:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This class cannot be instantiated.")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        try:
            kwargs["trust_remote_code"] = True
            return AutoModel.from_pretrained(*args, **kwargs)
        except (RequestException, HfHubHTTPError) as e:
            logger.warning(f"Online load failed ({e}), switching to offline mode...")
            kwargs["local_files_only"] = True
            return AutoModel.from_pretrained(*args, **kwargs)

class AutoModelForCausalLMSafeLoader: 
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This class cannot be instantiated.")
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        try:
            kwargs["trust_remote_code"] = True
            return AutoModelForCausalLM.from_pretrained(*args, **kwargs)
        except (RequestException, HfHubHTTPError) as e:
            logger.warning(f"Online load failed ({e}), switching to offline mode...")
            kwargs["local_files_only"] = True
            return AutoModelForCausalLM.from_pretrained(*args, **kwargs)

class AutoTokenizerSafeLoader:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This class cannot be instantiated.")
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        try:
            kwargs["trust_remote_code"] = True
            return AutoTokenizer.from_pretrained(*args, **kwargs)
        except (RequestException, HfHubHTTPError) as e:
            logger.warning(f"Online load failed ({e}), switching to offline mode...")
            kwargs["local_files_only"] = True
            return AutoTokenizer.from_pretrained(*args, **kwargs)
        
class AutoConfigSafeLoader:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This class cannot be instantiated.")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        try:
            kwargs["trust_remote_code"] = True
            return AutoConfig.from_pretrained(*args, **kwargs)
        except (RequestException, HfHubHTTPError) as e:
            logger.warning(f"Online load failed ({e}), switching to offline mode...")
            kwargs["local_files_only"] = True
            return AutoConfig.from_pretrained(*args, **kwargs)

class LoraConfigSafeLoader:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This class cannot be instantiated.")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        try:
            kwargs["trust_remote_code"] = True
            return LoraConfig.from_pretrained(*args, **kwargs)
        except (RequestException, HfHubHTTPError) as e:
            logger.warning(f"Online load failed ({e}), switching to offline mode...")
            kwargs["local_files_only"] = True
            return LoraConfig.from_pretrained(*args, **kwargs)

class LLMSafeLoader:
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This class cannot be instantiated.")

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        original_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        
        try:
            kwargs["trust_remote_code"] = True
            model = LLM(*args, **kwargs)
        except (RequestException, HfHubHTTPError) as e:
            logger.warning(f"Online load failed ({e}), switching to offline mode...")
            os.environ["HF_HUB_OFFLINE"] = "1"
            model = LLM(*args, **kwargs)
        finally:
            if original_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_hf_offline
                
        return model
            
            
