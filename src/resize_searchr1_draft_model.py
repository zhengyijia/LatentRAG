import os
import logging
import argparse

from llm_tools.safe_loader import (
    AutoModelForCausalLMSafeLoader,
    AutoModelSafeLoader, 
    AutoTokenizerSafeLoader, 
    AutoConfigSafeLoader, 
    LoraConfigSafeLoader
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Resize the embedding layer of the draft model to match the vocab size of the reference model (SearchR1).")
    parser.add_argument("--model_tag", type=str, required=True, help="Model tag for the draft model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path for draft model")
    parser.add_argument("--reference_model_tag", type=str, required=True, help="Model tag for the reference model")
    parser.add_argument("--reference_model_name_or_path", type=str, required=True, help="Model name or path for reference model")
    parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")

    args = parser.parse_args()

    # load model
    search_r1_tokenizer = AutoTokenizerSafeLoader.from_pretrained(
        args.reference_model_name_or_path,
        use_fast=True
    )
    draft_model_tokenizer = AutoTokenizerSafeLoader.from_pretrained(
        args.model_name_or_path,
        use_fast=True
    )

    assert search_r1_tokenizer.vocab_size == draft_model_tokenizer.vocab_size, \
        f"Tokenizer vocab size mismatch: {search_r1_tokenizer.vocab_size} vs {draft_model_tokenizer.vocab_size}"
    

    search_r1_config = AutoConfigSafeLoader.from_pretrained(
        args.reference_model_name_or_path
    )
    draft_model = AutoModelForCausalLMSafeLoader.from_pretrained(
        args.model_name_or_path,
        device_map="auto"
    )
    vocab_size = search_r1_config.vocab_size
    draft_model.resize_token_embeddings(vocab_size)

    draft_model.save_pretrained(args.save_path)
    draft_model_tokenizer.save_pretrained(args.save_path)

    logger.info(f"Resized draft model saved to {args.save_path}")

if __name__ == "__main__":
    main()


