import os
import json
import argparse
import spacy
from spacy.util import is_package, get_package_path
import subprocess
import importlib.util

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

model_name = "en_core_web_sm"

try:
    nlp = spacy.load(model_name)
except OSError:
    logger.info(f"Model '{model_name}' not found. Downloading...")
    subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
    nlp = spacy.load(model_name)


def parse_prompt(file_path: str, has_context=True):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content by the metadata pattern
    parts = content.split('# METADATA: ')
    parsed_data = []
    for part in parts[1:]:  # Skip the first split as it will be empty
        metadata_section, rest_of_data = part.split('\n', 1)
        metadata = json.loads(metadata_section)
        document_sections = rest_of_data.strip().split('\n\nQ: ')
        retrieved = document_sections[0].split('Wikipedia Title: ')[1:]
        retrieved = [chunk.strip() for chunk in retrieved]
        qa_pair = document_sections[1].split('\nA: ')
        question = qa_pair[0].strip()
        thought_and_answer = qa_pair[1].strip().split('So the answer is: ')
        thoughts = thought_and_answer[0].strip()
        answer = thought_and_answer[1].strip()

        # split sentences in thoughts
        # thoughts = [sent.text_with_ws for sent in nlp(thoughts).sents]
        thoughts = [sent.text for sent in nlp(thoughts).sents]

        parsed_data.append({
            'metadata': metadata,
            'retrieved': retrieved,
            'question': question,
            'thoughts': thoughts,
            'answer': answer
        })

    return parsed_data

def main(args):
    data_path = os.path.join('src', 'prompts', 'ircot_prompts', args.dataset, 'gold_with_3_distractors_context_cot_qa_codex.txt')
    save_path = os.path.join('src', 'prompts', 'ircot_prompts', args.dataset, 'gold_with_3_distractors_context_cot_qa_codex.json')
    parsed_data = parse_prompt(data_path, has_context=True)
    with open(save_path, 'w') as f:
        json.dump(parsed_data, f, indent=4)
    logger.info(f"Parsed data saved to {save_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    main(args)
