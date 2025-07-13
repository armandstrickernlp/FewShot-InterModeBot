# """ evlauate instruction finetuned MinTL"""
import argparse
import os
import json
import copy
import re
import pprint as pp
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          set_seed, 
                          logging, 
                          StoppingCriteria, 
                          StoppingCriteriaList
                            )
from peft import PeftModel
import torch

from mt_finetune.finetune_data.normalize_slot_values import normalize_state_slot_value
from mt_finetune.finetune_data.db_ops import MultiWozDB

from mwzeval.metrics import Evaluator as MWEvaluator




class ModelLoader:
    def __init__(self, model_name, cache_dir, checkpoint_path):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.checkpoint_path = checkpoint_path

    def load_base(self):
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map="auto",
        )
        # tokenizer = self.load_tokenizer()
        # base_model.resize_token_embeddings(len(tokenizer))
        return base_model

    def load_peft_model(self, base_model):
        peft_model = PeftModel.from_pretrained(
            base_model,
            self.checkpoint_path,
            device_map="auto",
            is_training=False,
        )
        return peft_model
    
    def load_tokenizer(self):
        tokenizer =  AutoTokenizer.from_pretrained(self.checkpoint_path)
        tokenizer.padding_side = "left"
        return tokenizer

class GenerationUtils:
    
    def get_state_dict(self, gen, dial_num):
        """get state from model output for inference"""
        try:
            flat_state = gen.split('<belief>')[1].split('</belief>')[0].strip()
        except IndexError as e:
            flat_state = ''
        state_dict = {}
        active_domains = set()
        triplets = flat_state.split(',')
        for triplet in triplets:
            triplet = triplet.split()
            try :
                domain, slot, val = triplet[0], triplet[1], ' '.join(triplet[2:])
            except:
                continue
            active_domains.add(domain)
            if slot == 'book':
                vals = val.split()
                slot, val = vals[0], ' '.join(vals[1:])
                
            if domain not in state_dict:
                state_dict[domain] = {slot: val}
            else:
                state_dict[domain].update({slot: val})
        if not active_domains:
            active_domains.add('general')
        return state_dict, active_domains, f"<belief> {flat_state} </belief>"

    def parse_resp(self, response):
        pattern = r"<response>(.*?)</response>"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            pattern = r"(?:Response:|Assistant:)\s*(.*?)(?=\n|$)"
            match = re.search(pattern, response, re.IGNORECASE)
            return match.group(1).strip() if match else response.strip()

    def llm_batch_generate(self, input_str, model, tokenizer, GEN_CONFIG): # stop_tokens in config, cant do batching though
        encoding = tokenizer(input_str, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                attention_mask=encoding.attention_mask,
                **GEN_CONFIG,
                )            
        generated_text = tokenizer.batch_decode(outputs[:, encoding.input_ids.shape[1]:], skip_special_tokens=False) # batched
        return generated_text


class StateTrackingUtils:
    def __init__(self, db_path):
        self.db = MultiWozDB(db_path=db_path)
        self.db_state_tokens = [['[db_state0]','[db_state1]','[db_state2]','[db_state3]','[db_state4]'], 
                    ['[db_state0+bookfail]', '[db_state1+bookfail]','[db_state2+bookfail]','[db_state3+bookfail]','[db_state4+bookfail]'],
                    ['[db_state0+booksuccess]','[db_state1+booksuccess]','[db_state2+booksuccess]','[db_state3+booksuccess]','[db_state4+booksuccess]']
                    ]
        self.domain_list = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "police"]
    
    def get_DB_token(self, constraint_dict, booking_api, current_domain):
        if current_domain != 'general':
            try: 
                constraint_dict[current_domain]
            except KeyError as e:
                print('Db error: current domain not in constraint dict')
                print(constraint_dict)
                print(current_domain)   
                return '[db_state0]', 0
            for slot in constraint_dict[current_domain]:
                # normalize to improve likeliness of finding a match in the db
                constraint_dict[current_domain][slot] = normalize_state_slot_value(slot, constraint_dict[current_domain][slot])

        matnums = self.db.get_match_num(constraint_dict)
        match = matnums[current_domain]
        dbvec = self.db.addDBPointer(current_domain, match)
        try:
            db_state = dbvec.index(1) +1
        except ValueError:
            db_state = 0

        # GOLD Booking api as this is not in the given db
        if booking_api == 'Success':
            bk_state = 2
        elif booking_api == 'Fail':
            bk_state = 1
        elif booking_api == 'None':
            bk_state = 0    
        return self.db_state_tokens[bk_state][db_state], match
    
    def update_seen(self, seen_domains, current_state):
        for domain in list(current_state.keys()):
            if domain not in seen_domains and domain in self.domain_list:
                seen_domains.append(domain)
        # handle errors where previous seen domains are not all in next state
        for domain in seen_domains:
            if domain not in current_state and domain != 'general':
                seen_domains.remove(domain)    
        return seen_domains  
 

class PredictionGenerator:
    def __init__(self, eval_examples, model, tokenizer, GEN_CONFIG, db_path):
        self.gen_utils = GenerationUtils() 
        self.st_utils = StateTrackingUtils(db_path=db_path)
        self.eval_examples = eval_examples
        self.model = model
        self.tokenizer = tokenizer
        self.GEN_CONFIG = GEN_CONFIG
     
    def format_msg(self, turn):
        messages = [
                    {'role': 'system', 'content': turn['sys_msg']}, 
                    {'role': 'user', 'content': turn['input']},
                ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return formatted
        
    def save_preds(self, predictions, predictions_path):
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=3)
    
    def generate_predictions(self, eval_examples, pred_path, break_idx=None):
        predictions = {}
        for idx, dial_num in tqdm(enumerate(eval_examples)):
            if idx == break_idx:
                break
            # if idx in range(1):
            #     continue
            batch_bs = []
            seen_domains = ['general']
            preds_over_turns = []
            for turn in eval_examples[dial_num]:
                formatted_msg = self.format_msg(turn)
                batch_bs.append(formatted_msg)
            # pp.pprint(batch_bs)

            # generate belief state for db search
            generated_bs = self.gen_utils.llm_batch_generate(batch_bs, self.model, self.tokenizer, self.GEN_CONFIG)
            # pp.pprint(generated_bs)
        
            batch_resp = []
            for turn, context, gen in zip(eval_examples[dial_num], batch_bs, generated_bs):
                state_dict, active_domains, flat_belief = self.gen_utils.get_state_dict(gen, dial_num)
                seen_domains = self.st_utils.update_seen(seen_domains, state_dict)
                current_domain = seen_domains[-1]
                
                # if state_dict != turn['gold_state']:
                #     with open('errors.txt', 'a') as f:
                #         f.write(dial_num+'\n')
                #         f.write('State mismatch\n')
                #         f.write(turn['context']+'\n')
                #         f.write(turn['user_request']+'\n')
                #         f.write(f'Predicted State: {state_dict}'+'\n')
                #         f.write(f'Gold State: {turn["gold_state"]}'+'\n')
                #         f.write('\n')
                
                # get db_state
                db_state, _ = self.st_utils.get_DB_token(state_dict, turn['booking_api'], current_domain)
                msg_to_complete = f"{context}{flat_belief} <db> {db_state} </db>"
                batch_resp.append(msg_to_complete)
                
                preds_over_turns.append({'state': state_dict, 'active_domains': [current_domain]})
            
            # pp.pprint(batch_resp)
        
            # generate response
            generated_resp = self.gen_utils.llm_batch_generate(batch_resp, self.model, self.tokenizer, self.GEN_CONFIG)
            for gen, pred in zip(generated_resp, preds_over_turns):
                resp_str = self.gen_utils.parse_resp(gen)
                pred['response'] = resp_str
            
            predictions[dial_num.lower()] = preds_over_turns
            # pp.pprint(predictions)
            self.save_preds(predictions, pred_path)
        return predictions

def map_results_to_gold(results):
    mapping = {
        "restaurant":
                {
                    "day": "bookday",
                    "time": "booktime",
                    "people": "bookpeople",
                },
        "hotel":
                {
                    "stay": "bookstay",
                    "day": "bookday",
                    "people": "bookpeople",
                },
        "train":
                {
                    "people": "bookpeople",
                },    
            }
    results_mapped = {}
    for dial_num in results:
        turns = []
        for turn in results[dial_num]:
            turn_dict = {}
            new_state = {}
            for domain in turn["state"]:
                if domain in mapping:
                    new_state[domain] = {}
                    for slot, val in turn["state"][domain].items():
                        if slot in mapping[domain]:
                            new_state[domain][mapping[domain][slot]] = val
                        else:
                            new_state[domain][slot] = val
                else:
                    new_state[domain] = turn["state"][domain]
            turn_dict['active_domains'] = turn['active_domains']
            turn_dict['response'] = turn['response']
            turn_dict['state'] = new_state
            turns.append(turn_dict)
        results_mapped[dial_num] = turns
    return results_mapped  

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--cache_dir', type=str, default='./model_cache')
    parser.add_argument('--checkpoint_path', type=str, default="mt_finetune/checkpoints/5e-05_42_rank64_q_k_v_o/test_checkpoint")
    parser.add_argument("--eval_set_path", type=str, default="mt_finetune/finetune_data/eval_data/mwoz")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument('--db_path', type=str, default='mt_finetune/finetune_data/db')
    parser.add_argument('--max_new_tokens', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do_sample', type=bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)

    # load model and tokenizer
    model_loader = ModelLoader(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        checkpoint_path=args.checkpoint_path
    )   
    base_model = model_loader.load_base()
    peft_model = model_loader.load_peft_model(base_model)
    tokenizer = model_loader.load_tokenizer()

    # load data and prep output_dir
    with open(os.path.join(args.eval_set_path, f"{args.eval_split}.json")) as f: 
        eval_examples = json.load(f)

    output_dir = os.path.join(
        "mt_results",
        f"{args.eval_set_path.split('/')[-1]}_{args.checkpoint_path.split('/')[-2]}", # 8e-05_42
    )
    os.makedirs(output_dir, exist_ok=True)
    pred_path = os.path.join(output_dir, f"generated_{args.eval_split}.json")

    # run inference
    GEN_CONFIG = {
    'max_new_tokens': args.max_new_tokens,
    'do_sample': args.do_sample,
    'num_return_sequences': 1,
    'pad_token_id' : tokenizer.eos_token_id,
    'no_repeat_ngram_size': 10
    } 

    pg = PredictionGenerator(eval_examples, peft_model, tokenizer, GEN_CONFIG, db_path=args.db_path)
    results = pg.generate_predictions(eval_examples, pred_path, break_idx=None)

    # map results to gold slot names
    results = map_results_to_gold(results)

    evaluator = MWEvaluator(bleu=True, success=True, richness=True, jga=True, dst=True, filter=True)
    eval_results = evaluator.evaluate(results)

    os.makedirs(os.path.join(output_dir, 'eval'), exist_ok=True)
    with open(os.path.join(output_dir, 'eval', f"eval_{args.eval_split}"), 'w') as f:
        json.dump(eval_results, f, indent=2)




   

            





    



    
