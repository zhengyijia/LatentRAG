import os
import argparse
import json
import ijson
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate browsecomp responses using Qwen judge model."
    )
    parser.add_argument(
        "--input_path", required=True, help="Directory containing run JSON files"
    )
    parser.add_argument(
        "--filter_em_correct", action="store_true", help="Whether to filter reasoning steps with EM score of 1.0"
    )
    parser.add_argument(
        "--filter_llm_judge_correct", action="store_true", help="Whether to filter reasoning steps judged as correct by LLM judge"
    )
    parser.add_argument(
        "--delete_input_file", action="store_true", help="Whether to delete input file after evaluation"
    )
    args = parser.parse_args()
    
    if not args.filter_em_correct and not args.filter_llm_judge_correct:
        print("No filtering option selected. Exiting.")
        return
    
    param_tag = ''
    if args.filter_em_correct:
        param_tag += '_em_correct'
    if args.filter_llm_judge_correct:
        param_tag += '_llm_judge_correct'
    
    save_dir = os.path.dirname(args.input_path)
    save_filename = os.path.basename(args.input_path).replace(".json", f"{param_tag}.json")
    args.save_path = os.path.join(save_dir, save_filename)
    
    # with open(args.input_path, 'r') as f:
    #     results = json.load(f)
    
    # if args.filter_em_correct:
    #     results['results'] = [item for item in results['results'] if item['em_score']==1.0]
        
    # if args.filter_llm_judge_correct:
    #     results['results'] = [item for item in results['results'] if item['llm_judge_result']['correct']==True]
    
    results = {}
    with open(args.input_path, "rb") as f:
        print(f"Start reading {args.input_path}")
        for k, v in ijson.kvitems(f, "", use_float=True):
            if k == "results":
                continue
            results[k] = v
        
    results['results'] = []
    with open(args.input_path, "rb") as f:
        for item in tqdm(ijson.items(f, "results.item", use_float=True)):
            if args.filter_em_correct and item["em_score"] != 1.0:
                continue

            if args.filter_llm_judge_correct and not item["llm_judge_result"]["correct"]:
                continue

            results['results'].append(item)
        
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved filtered results to: {args.save_path}")
        
    if args.delete_input_file:
        os.remove(args.input_path)
        print(f"Deleted input file: {args.input_path}")
    

if __name__ == "__main__":
    main()

