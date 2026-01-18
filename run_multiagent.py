import os
import json
import time
import re
import argparse
from typing import Dict, List, Any, Type, Set
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM

from src.mirage.coordinator import MultiAgentCoordinator
from config import get_config, update_config_from_args

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder with support for set serialization"""
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def json_serialize(obj):
    """Recursively convert sets to lists for JSON compatibility"""
    if isinstance(obj, dict):
        return {k: json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

def extract_question_from_item(item):
    """Extract question from data item
    
    Supports formats:
    1. {"input": "question"} 
    2. {"qustion_output": "<CLS>question<SEP>entities<EOS>"}
    """
    if 'input' in item:
        return item['input']

    if 'qustion_output' in item:
        qustion_output = item['qustion_output']
        if '<CLS>' in qustion_output and '<SEP>' in qustion_output:
            start_idx = qustion_output.find('<CLS>') + 5
            end_idx = qustion_output.find('<SEP>')
            if start_idx < end_idx:
                return qustion_output[start_idx:end_idx].strip()

    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Agent Medical QA System")

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--question', type=str, help="Single question for interactive testing")
    run_mode.add_argument('--dataset_path', type=str, help="Dataset path for batch processing")

    parser.add_argument('--subset_num', type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument('--model_path', type=str, help="Model path")
    parser.add_argument('--neo4j_uri', type=str, default="bolt://127.0.0.1:17687", help="Neo4j URI")
    parser.add_argument('--neo4j_user', type=str, default="neo4j", help="Neo4j username")
    parser.add_argument('--neo4j_password', type=str, help="Neo4j password")
    parser.add_argument('--max_sub_questions', type=int, default=4, help="Maximum sub-questions per question")
    parser.add_argument('--max_retries', type=int, default=3, help="Maximum retries per sub-question")

    return parser.parse_args()

def main():
    args = parse_args()
    update_config_from_args(args)
    config = get_config()

    question = args.question
    dataset_path = config.data.default_dataset_path
    subset_num = config.data.subset_num

    if question:
        print('-----------------------')
        print(f'Interactive mode: Single question')
        print(f'Question: {question}')
        print('-----------------------')
    else:
        print('-----------------------')
        print(f'Dataset mode: {dataset_path}')
        print('-----------------------')

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    llm = LLM(
        model=config.model.model_path,
        tensor_parallel_size=config.model.tensor_parallel_size,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
    ) 
    print("Model loaded!")

    coordinator = MultiAgentCoordinator(
        llm=llm,
        tokenizer=tokenizer,
        neo4j_uri=config.neo4j.uri,
        neo4j_user=config.neo4j.user,
        neo4j_password=config.neo4j.password
    )

    if question:
        print("Processing question...")
        start_time = time.time()
        result = coordinator.process_question(
            question,
            max_sub_questions=config.search.max_sub_questions,
            max_retries=config.search.max_retries
        )
        end_time = time.time()
        
        print("\n" + "="*80)
        print("Statistics:")
        print(f"- Time: {end_time - start_time:.2f}s")
        print(f"- Searches: {result['metrics']['total_searches']}")
        print(f"- Validations: {result['metrics']['successful_validations']}/{result['metrics']['failed_validations']}")
        
        print("\nFinal Answer:")
        print("="*80)
        print(result["final_answer"])
        print("="*80)
    
    else:
        filtered_data = []
        
        print("Loading dataset...")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        filtered_data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return

        if subset_num > 0:
            filtered_data = filtered_data[:subset_num]
            print(f"Processing {subset_num} samples")
        else:
            print(f"Processing all {len(filtered_data)} samples")
        
        print("\nQuestions to process:")
        for i, item in enumerate(filtered_data):
            question = extract_question_from_item(item)
            if question:
                print(f"Q{i+1}: {question[:50]}..." if len(question) > 50 else f"Q{i+1}: {question}")
            else:
                print(f"Q{i+1}: <invalid format>")
        print("")
        
        results = []
        start_time = time.time()
        
        for item in tqdm(filtered_data, desc="Processing", ncols=100):
            question = extract_question_from_item(item)
            if not question:
                tqdm.write(f"⚠️ Skipped: Unable to extract question")
                continue
            
            result = coordinator.process_question(
                question,
                max_sub_questions=config.search.max_sub_questions,
                max_retries=config.search.max_retries
            )
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✅ Completed! Total: {total_time:.2f}s | Average: {total_time/len(results):.2f}s/question")


if __name__ == "__main__":
    main()
