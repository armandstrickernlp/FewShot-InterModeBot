import argparse
import json
from tqdm import tqdm
import os
import pprint as pp
import collections

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from model import (
    ZeroShotOpenAIChatLLM,
    ZeroShotLlama,
)

from loaders import load_mwoz
from definitions.chitchat import  task_selection_prompt
from utils import parse_task

from sklearn.metrics import classification_report


def str_to_bool(s):
    if s.lower() in ['true', 't', 'yes', 'y']:
        return True
    elif s.lower() in ['false', 'f', 'no', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--dataset_variant", type=str, default='fused', help="1 of mwoz, interference, or fused")
    parser.add_argument("--context_size", type=int, default=3) # number of prev turns to consider for retrieval
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--single_domain", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--restrict_domains", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='./model_cache')
    parser.add_argument("--output_dir", type=str, default='results/llama8B')
    args = parser.parse_args()

    print(args)

    if args.model_name.startswith("gpt-"):
        classif_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=False)

    elif args.model_name.startswith("meta-llama"):
        loaded_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=args.cache_dir,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        classif_model = ZeroShotLlama(loaded_model, tokenizer)
    
    if args.dataset_variant == 'fused':
        with open(os.path.join(args.output_dir, "results_fused_func51.json")) as f:
            results = json.load(f)
    elif args.dataset_variant == 'interference':
        with open(os.path.join(args.output_dir, "results_interference_func51.json")) as f:
            results = json.load(f)

    
    data_gen = load_mwoz(args.database_path, # load turns
                    args.context_size, 
                    split=args.split, 
                    total=500, 
                    shuffle=False, 
                    only_single_domain=args.single_domain, 
                    restrict_domains=args.restrict_domains.split(",") if args.restrict_domains is not None else None,
                    variant=args.dataset_variant
                    )
    
 

    if args.dataset_variant == 'fused':
        with open("mt_finetune/data/eval_data/fused/test.json") as f:
            data = json.load(f)
    elif args.dataset_variant == 'interference':
        with open("data/interference_data/test.json") as f:
            data = json.load(f)
    
    
    preds = collections.defaultdict(list)
    gold = collections.defaultdict(list)

    if args.dataset_variant == 'fused':
        prev_dial_num = None
        for turn in tqdm(data_gen):
            if turn["dialogue_id"] != prev_dial_num:
                prev_dial_num = turn["dialogue_id"]

                context_list = turn['metadata']['context_list']

                for i in range(0, len(context_list), 2):
                    question = context_list[i]
                    history = "\n".join(context_list[i-2:i])
                    task, dp = classif_model(task_selection_prompt, predict=True, history=history, utterance=f"{question.strip()}")
                    selected_task = parse_task(task)
                    preds[turn["dialogue_id"]].append(selected_task)
                    gold[turn["dialogue_id"]].append('chitchat')

                
                for idx, (turn_pred, turn_gold) in enumerate(zip(results[turn["dialogue_id"]],  data[turn["dialogue_id"].upper()])):
                    preds[turn['dialogue_id']].append('task') if turn_pred['domain'] != 'general' else preds[turn['dialogue_id']].append('chitchat')
                    if turn_pred['domain'] == 'general' and (idx == len(results[turn['dialogue_id']]) - 1 or idx == len(results[turn['dialogue_id']]) - 2 or "thank" in turn_pred['question'].lower() or "bye" in turn_pred['question'].lower()):
                        gold[turn['dialogue_id']].append('chitchat')
                    else:
                        gold[turn['dialogue_id']].append('task')
    

    elif args.dataset_variant == 'interference':
        preds_backstory_turn = []
        gold_backstory_turn = []
        for dial_num in results:
            backstory_turn_idx = data[dial_num.upper()]['augmented_idx'][0]
            for idx, turn in enumerate(results[dial_num]):
                preds[dial_num].append('task') if turn['domain'] != 'general' else preds[dial_num].append('chitchat')
                if turn['domain'] == 'general' and (idx == len(results[dial_num]) - 1 or idx == len(results[dial_num]) - 2 or "thank" in turn['question'].lower() or "bye" in turn['question'].lower()):
                    gold[dial_num].append('chitchat')
                else:
                    gold[dial_num].append('task')
                
                if idx == backstory_turn_idx//2:
                    preds_backstory_turn.append('task') if turn['domain'] != 'general' else preds_backstory_turn.append('chitchat')
                    if turn['domain'] == 'general' and (idx == len(results[dial_num]) - 1 or idx == len(results[dial_num]) - 2 or "thank" in turn['question'].lower() or "bye" in turn['question'].lower()):
                        gold_backstory_turn.append('chitchat')
                    else:
                        gold_label = 'task'
                        gold_backstory_turn.append(gold_label)
                    if turn['domain'] == 'general' and gold_label == 'task':
                        print(turn['question'])
                        print(turn['response'])
                        print()
            

    # flatten dictionary lists, make sure its the same order
    preds = sorted([(k, v) for k, v in preds.items()])
    gold = sorted([(k, v) for k, v in gold.items()])


    # flatten lists
    preds_flat = []
    for l in preds:
        preds_flat.extend(l[1])
    gold_flat = []
    for l in gold:
        gold_flat.extend(l[1])
    
    scores = classification_report(gold_flat, preds_flat)

    # write to file
    os.makedirs(os.path.join(args.output_dir, 'eval_classif_f1'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'eval_classif_f1', f"classif_f1_{args.dataset_variant}.txt"), 'w') as f:
        f.write(scores)
    
    if args.dataset_variant == 'interference':
        scores_backstory_turn = classification_report(gold_backstory_turn, preds_backstory_turn)
        with open(os.path.join(args.output_dir, 'eval_classif_f1', "classif_f1_backstory_turn.txt"), 'w') as f:
            f.write(scores_backstory_turn)

    
            
            
      

    