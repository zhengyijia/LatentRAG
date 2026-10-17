import os
import argparse
import jsonlines

parser = argparse.ArgumentParser(description="Convert boolean answers to string answers in a dataset. True -> 'yes', False -> 'no'.")
parser.add_argument("--data_path", type=str, default=None, help="path to the dataset folder")
parser.add_argument("--data_file", type=str, default=None, help="dataset file subpath")

args = parser.parse_args()

with jsonlines.open(os.path.join(args.data_path, args.data_file), mode='r') as reader:
    data = [obj for obj in reader]
    
for item in data:
    item['golden_answers'] = ['yes' if ans is True else 'no' if ans is False else ans for ans in item['golden_answers']]

with jsonlines.open(os.path.join(args.data_path, args.data_file), mode='w') as writer:
    writer.write_all(data)
    
print(f"Converted boolean answers to string answers in {os.path.join(args.data_path, args.data_file)}")
