import os
import argparse
import torch
import json
import re

from llm_tools.vllm_offline import VLLMOffline
from llm_tools.safe_loader import AutoTokenizerSafeLoader
from vllm import LLM, SamplingParams

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()

# GRADER_TEMPLATE = """
# Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.
# The [response] is placed between <response> and </response> tags.

# [question]: {question}

# [response]: 
# <response>
# {response}
# </response>

# [correct_answer]: {correct_answer}

# Your judgement must be in the format and criteria specified below:

# extracted_final_answer: The final exact answer extracted from the [response]. 

# [correct_answer]: Repeat the [correct_answer] given above.

# reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

# correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

# confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
# """.strip()

def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer (try bold formats first, then regular)
    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    # Extract reasoning/explanation
    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract correct (yes/no)
    correct_match = re.search(
        r"\*\*correct:\*\*\s*\*{0,2}\s*(yes|no)\s*\*{0,2}", judge_response, re.IGNORECASE
    )
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*\*{0,2}\s*(yes|no)\s*\*{0,2}", judge_response, re.IGNORECASE
        )
    if not correct_match:
        correct_match = re.search(r"correct:\s*\*{0,2}\s*(yes|no)\s*\*{0,2}", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    # Extract confidence (percentage)
    confidence_match = re.search(
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
    )
    if not confidence_match:
        confidence_match = re.search(
            r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if not confidence_match:
        confidence_match = re.search(
            r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", judge_response, re.IGNORECASE
        )
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
        if result["confidence"] > 100:
            result["confidence"] = 100

    # Check if we got the essential fields
    if result["correct"] is None:
        result["parse_error"] = True

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate browsecomp responses using Qwen judge model."
    )
    parser.add_argument(
        "--input_path", required=True, help="Directory containing run JSON files"
    )
    parser.add_argument(
        "--model_name_or_path", default="Qwen/Qwen3-32B", help="Qwen model for judging"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Decoding temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8, help="Top-p nucleus sampling"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Top-k sampling (sent via extra_body)"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=4096,
        help="Maximum output tokens for judge model",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum total tokens for judge model",
    )
    parser.add_argument(
        "--max_num_seqs", type=int, default=64, help="Maximum number of sequences for vLLM inference"
    )
    parser.add_argument(
        "--delete_input_file", action="store_true", help="Whether to delete input file after evaluation"
    )
    args = parser.parse_args()
    
    save_dir = os.path.dirname(args.input_path)
    save_filename = os.path.basename(args.input_path).replace(".json", "_judged.json")
    args.save_path = os.path.join(save_dir, save_filename)
    
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizerSafeLoader.from_pretrained(
        args.model_name_or_path,
        use_fast=True
    )
    llm = LLM(model=args.model_name_or_path, 
              tensor_parallel_size=torch.cuda.device_count(), 
              pipeline_parallel_size=1,
              seed=42,
              gpu_memory_utilization=0.85,
              max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, 
              dtype=torch_dtype,)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_output_tokens,
    )
    
    with open(args.input_path, 'r') as f:
        results = json.load(f)
    
    all_prompts = []
    for item in results['results']:
        # Only use the first golden answer
        answer = item['answer'].strip()
        # remove special tags from the answer
        answer = answer.replace("<thinking>", " ").replace("</thinking>", " ")
        answer = answer.replace("<subquery>", " ").replace("</subquery>", " ")
        answer = answer.replace("<answer>", " ").replace("</answer>", " ")
        answer = answer.replace("...(truncated)", "")
        answer = answer.strip()
        answer = "No answer provided." if not answer else answer

        all_prompts.append(GRADER_TEMPLATE.format(question=item['query'], response=answer, correct_answer=item['golden_answers'][0]))

    messages_list = [[{"role": "user", "content": p}] for p in all_prompts]
    outputs = llm.chat(
        messages_list,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    all_correct_scores = []
    num_parse_errors = 0
    for item, out in zip(results['results'], outputs or []):
        judge_text = ""
        try:
            if getattr(out, "outputs", None):
                judge_text = out.outputs[0].text or ""
        except Exception:
            judge_text = ""

        judge_result = parse_judge_response(judge_text)
        item['llm_judge_response'] = judge_text
        item['llm_judge_result'] = judge_result 

        if judge_result.get("parse_error", False):
            print(f"Parse error: {judge_text}")
            num_parse_errors += 1
        else:
            all_correct_scores.append(judge_result.get("correct", False))
            if 'reference_llm_judge_result' in item and item['reference_llm_judge_result'] is not None:
                current_model_correct = int(judge_result.get("correct", None))
                reference_correct = int(item['reference_llm_judge_result'].get("correct", None))
                if reference_correct is not None:
                    item['llm_judge_comparison_with_reference'] = f'llm_judge_cur_{current_model_correct}_reference_{reference_correct}'
    
    llm_judge_accuracy = sum(all_correct_scores) / len(all_correct_scores) if all_correct_scores else 0.0
    print(f"Judging completed. Accuracy: {llm_judge_accuracy:.4f}")
    if num_parse_errors > 0:
        print(f"Warning: Number of parse errors: {num_parse_errors}!!!!!!!!", flush=True)
    else:
        print("No parse errors detected in judge responses.", flush=True)
    
    results = {
        'llm_judge_model': args.model_name_or_path,
        'llm_judge_accuracy': llm_judge_accuracy, 
        'num_llm_judge_parse_errors': num_parse_errors,
        **results,
    }
    
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved judged results to: {args.save_path}")
        
    if args.delete_input_file:
        os.remove(args.input_path)
        print(f"Deleted input file: {args.input_path}")
        

if __name__ == "__main__":
    main()
