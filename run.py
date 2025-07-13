import argparse
import pickle
import json
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import logging
import transformers
import random
import torch
import copy

import os
import pprint as pp


from model import (
    FewShotOpenAIChatLLM,
    ZeroShotOpenAIChatLLM,
    FewShotLlama,
    ZeroShotLlama,
    FewShotGorilla,
    ZeroShotGorilla
    )

from loaders import load_mwoz
from delex import prepareSlotValuesIndependent, delexicalise, delexicaliseReferenceNumber

from definitions.base import MW_FEW_SHOT_DOMAIN_DEFINITIONS, MW_ZERO_SHOT_DOMAIN_DEFINITIONS, multiwoz_domain_prompt
from definitions.chitchat import chitchat_prompt
from definitions.selection import task_selection_prompt, domain_selection_prompt
from definitions.function_call import FC_FEW_SHOT_DOMAIN_DEFINITIONS, FC_ZERO_SHOT_DOMAIN_DEFINITIONS
from definitions.sql import SQL_FEW_SHOT_DOMAIN_DEFINITIONS

from database import MultiWOZDatabase
from utils import (parse_state, 
                   parse_func_call,  
                   parse_func_call_gpt,
                   parse_sql, 
                   parse_response, 
                   parse_domain,
                   parse_task, 
                   ExampleRetriever, 
                   ExampleFormatter
                )

from mwzeval.metrics import Evaluator as MWEvaluator

transformers.set_seed(42)


# Configure the logger
logging.basicConfig(
    filename='local_logs/output.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO   
)
logger = logging.getLogger()




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
    parser.add_argument("--support_model_name", type=str, default='')
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--num_examples", type=int, default=5) # number of 'shots'
    parser.add_argument("--dials_total", type=int, default=20) # total number of dialogues to run for testing
    parser.add_argument("--testing", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--dataset_variant", type=str, default='interference', help="1 of mwoz, interference, or fused")
    parser.add_argument("--context_size", type=int, default=3) # number of prev turns to consider for retrieval
    parser.add_argument("--ontology", type=str, default="ontology.json")
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--single_domain", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--restrict_domains", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default='./model_cache')
    parser.add_argument("--results_dir", type=str, default='results/llama8B')
    parser.add_argument("--print", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--function_call", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--sql", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--zero_shot_state", type=str_to_bool, choices=[True, False], default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(args)

    os.makedirs(args.results_dir, exist_ok=True)

    transformers.set_seed(args.seed)

    out_file_name = f'results_{args.dataset_variant}'
    if args.function_call:
        out_file_name += '_func'
    elif args.sql:
        out_file_name += '_sql'
    if args.zero_shot_state:
        out_file_name += '_zs'
    if args.support_model_name:
        if args.support_model_name.startswith("meta-llama"):
            out_file_name += f'_llama'
        elif args.support_model_name.startswith("gpt-"):
            out_file_name += f'_gpt'

    out_file_name += str(args.seed) + '.json'


    if args.model_name.startswith("gpt-"):
        if args.zero_shot_state: 
            state_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=True, seed=args.seed)
        else:
            state_model = FewShotOpenAIChatLLM(args.model_name, seed=args.seed)
        response_model = FewShotOpenAIChatLLM(args.model_name, seed=args.seed)
        domain_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=False, seed=args.seed)
        chitchat_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=False, seed=args.seed)
        
    elif args.model_name.startswith("meta-llama"):
        loaded_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=args.cache_dir,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        state_model, response_model = FewShotLlama(loaded_model, tokenizer), FewShotLlama(loaded_model, tokenizer)
        domain_model, chitchat_model = ZeroShotLlama(loaded_model, tokenizer), ZeroShotLlama(loaded_model, tokenizer)

    elif args.model_name.startswith("gorilla-llm"):
        loaded_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=args.cache_dir,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        llama_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=args.cache_dir,
            )
        llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir=args.cache_dir)
        if args.zero_shot_state:
            state_model = ZeroShotGorilla(loaded_model, tokenizer)
        else:
            state_model = FewShotGorilla(loaded_model, tokenizer) #  use gorilla only for state tracking
        if args.support_model_name.startswith("meta-llama"):
            response_model = FewShotLlama(llama_model, llama_tokenizer)
            domain_model = ZeroShotLlama(llama_model, llama_tokenizer)
            chitchat_model = ZeroShotLlama(llama_model, llama_tokenizer)
        elif args.support_model_name.startswith("gpt-"):
            response_model = FewShotOpenAIChatLLM(args.support_model_name)
            domain_model = ZeroShotOpenAIChatLLM(args.support_model_name, func_call=False)
            chitchat_model = ZeroShotOpenAIChatLLM(args.support_model_name, func_call=False)


    # load faiss db
    with open(args.faiss_db, 'rb') as f:
        faiss_vs = pickle.load(f)
    with open(args.ontology, 'r') as f:
        ontology = json.load(f)  
    
    # load utils
    database = MultiWOZDatabase(args.database_path)
    state_vs = faiss_vs
    delex_dic = prepareSlotValuesIndependent(args.database_path)

    example_retriever = ExampleRetriever(faiss_vs)
    state_retriever = ExampleRetriever(state_vs)
    example_formatter = ExampleFormatter(ontology=ontology)

    # run generation
    history = []
    n = 0
    results = {}
    results_wo_state = {}
    last_dial_id = None
    total = args.dials_total if args.testing else 500
    variant = args.dataset_variant if args.dataset_variant in ['interference', 'fused'] else None
    data_gen = load_mwoz(args.database_path, # load turns
                            args.context_size, 
                            split=args.split, 
                            total=total, 
                            shuffle=False, 
                            only_single_domain=args.single_domain, 
                            restrict_domains=args.restrict_domains.split(",") if args.restrict_domains is not None else None,
                            variant=variant
                            )
    tn = 0 # turn number
    progress_bar = tqdm(total=total, desc="Generating responses") # 3545
    for it, turn in enumerate(data_gen):
        if last_dial_id != turn['dialogue_id']:
            last_dial_id = turn['dialogue_id']
            n += 1
            progress_bar.update(1)
            tn = 0
            with open(os.path.join(args.results_dir, out_file_name), 'w') as f:
                json.dump(results, f, indent=2)
            if args.testing and n > args.dials_total:
                break
            history = []
            dialogue_id = turn['dialogue_id']
            results[dialogue_id] = []
            results_wo_state[dialogue_id] = []
            total_state = {}
            if args.print:
                print('=' * 100)
            previous_domain = None
        tn += 1

        # print(turn["dialogue_id"]) ######################################
        # if n <= 395:
        #     continue

        question = turn['question']
        print(f"QUESTION: {question}")
        context_list = turn['metadata']['context_list']
        context = turn['metadata']['context']
        gold_response = turn['metadata']['response']
        gt_state = turn['gt_state']
        gt_domain = turn['metadata']['domain']

        if tn == 1: # first turn of new dialogue ####
            history += context_list[:-1]

        if len(gt_state) == 0:
            gt_state = {}

        new_gt_state = {}
        for domain, ds in gt_state.items():
            for sl, val in ds.items():
                if domain not in new_gt_state:
                    new_gt_state[domain] = {sl: val}
                else:
                    new_gt_state[domain][sl] = val
        

        retrieve_history = context_list
        retrieved_examples = example_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=5)
        retrieved_domains = [example['domain'] for example in retrieved_examples]

        # predict turn type (task or chitchat)
        #  with ctxt (seed 42, 44)
        task, dp = domain_model(task_selection_prompt, predict=True, history=context, utterance=f"Customer: {question.strip()}")

        # without (43)
        # task, dp = domain_model(task_selection_prompt, predict=True, history='', utterance=f"Customer: {question.strip()}")
       
        print(f"RAW TASK: {task}")
        selected_task = parse_task(task)
        print(f"TASK: {selected_task}")
                
        if selected_task == 'chitchat':
            selected_domain = 'general' # for results evaluation
            # no update to state
            response, filled_prompt = chitchat_model(chitchat_prompt, predict=True, history="\n".join(context_list[:-1]), utterance=f"Customer: {question.strip()}")
            response_raw = response
            response = parse_response(response)
        
        else:
             # predict domain
            selected_domain, dp = domain_model(domain_selection_prompt, predict=True, history="\n".join(context_list[-3:-1]), utterance=f"Customer: {question.strip()}")
            selected_domain = parse_domain(selected_domain)
            print(f"DOMAIN: {selected_domain}")

            available_domains = [dom for dom in MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys() if dom != 'hospital']
            if args.print:
                print(f"PREDICTED DOMAIN: {selected_domain}")
                logger.info(f"PREDICTED DOMAIN: {selected_domain}")
            if selected_domain not in available_domains: # pick random if not in possible choices
                selected_domain = random.choice(available_domains)
            if previous_domain != selected_domain: # update prev domain
                previous_domain = selected_domain
            
            # load prompt class for predicted domain
            if args.function_call:
                if args.model_name.startswith("meta-llama") :
                    domain_definition = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
                    state_prompt = domain_definition.state_prompt
                    response_prompt = domain_definition.response_prompt
                    expected_slots = domain_definition.expected_slots
                
                elif  args.model_name.startswith("gorilla-llm"): 
                    if args.zero_shot_state:
                        state_prompt = FC_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain].state_prompt
                    else:
                        state_prompt = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].state_prompt
                    response_prompt = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].response_prompt
                    expected_slots = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].expected_slots # few or zero doesnt matter

                elif args.model_name.startswith("gpt-"):
                    if args.zero_shot_state:
                        state_prompt = FC_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain].state_prompt
                    else:
                        state_prompt = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].state_prompt
                    response_prompt = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].response_prompt # classic few shot response prompt
                    expected_slots = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].expected_slots # few or zero doesnt matter

            elif args.sql:
                if args.model_name.startswith("meta-llama"):
                    domain_definition = SQL_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
                    state_prompt = domain_definition.state_prompt
                    response_prompt = domain_definition.response_prompt
                    expected_slots = domain_definition.expected_slots
                    
            else:
                # use few shot for baselines
                domain_definition = MW_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
                state_prompt = domain_definition.state_prompt
                response_prompt = domain_definition.response_prompt
                expected_slots = domain_definition.expected_slots



            # make state and response prompts
            retrieved_examples = [example for example in retrieved_examples] #if example['domain']] == selected_domain] # maybe use all examples anyway?
            num_examples = min(len(retrieved_examples), args.num_examples)

            num_state_examples = args.num_examples
            # state_examples = [example for example in state_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20) if example['domain'] == selected_domain][:num_state_examples]
            state_examples = [example for example in state_retriever.retrieve("\n".join(retrieve_history[-args.context_size:]), k=20)][:num_state_examples]
            
            positive_state_examples = example_formatter.format(state_examples[:num_state_examples],
                                                                input_keys=["context"],
                                                                output_keys=["state"],
                                                                func_call=args.function_call,
                                                                write_sql=args.sql,
                                                                )

            # negative_state_examples = example_formatter.format(state_examples[:num_state_examples],
            #                                                     input_keys=["context"],
            #                                                     output_keys=["state"],
            #                                                     corrupt_state=True)
            
            response_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                            input_keys=["context", "full_state", "database"],
                                                            output_keys=["response"],
                                                            use_json=True)

        
            # PREDICT STATE
            try:
                kwargs = {
                    "history": '\n'.join(context_list[:-1]),
                    "utterance": question.strip()
                }
                if not args.zero_shot_state:
                    kwargs["positive_examples"] = positive_state_examples
                    kwargs["negative_examples"] = [] # negative_state_examples not added
                
                # PREDICTION
                
                state, filled_state_prompt = state_model(state_prompt, predict=True, **kwargs) # PREDICTION
                if n < 2 and args.print:
                    print("Filled prompt:", filled_state_prompt)
                    logger.info(f"Filled prompt: {filled_state_prompt}")
            except:
                state = "{}"
            
            
            # PARSE STATE
            if args.function_call:  
                if args.zero_shot_state:
                    if args.model_name.startswith("gorilla-llm"):
                        parsed_state = parse_func_call(state, default_domain=selected_domain) # same parsing as for few shot
                    elif args.model_name.startswith("gpt-"):
                        parsed_state = parse_func_call_gpt(state, default_domain=selected_domain)

                else:
                    parsed_state = parse_func_call(state, default_domain=selected_domain)
            
            
            elif args.sql:
                parsed_state = parse_sql(state, default_domain=selected_domain)
            else:
                parsed_state = parse_state(state, default_domain=selected_domain)


            if selected_domain not in parsed_state:
                parsed_state[selected_domain] = {}
            if not isinstance(parsed_state[selected_domain], dict):
                parsed_state[selected_domain] = {}
            keys_to_remove = [k for k in parsed_state[selected_domain].keys() if k not in expected_slots]
            for k in keys_to_remove:
                del parsed_state[selected_domain][k]
            try:
                for domain, ds in parsed_state.items():
                    for slot, value in ds.items():
                        pass
            except:
                parsed_state = {selected_domain: {}}
            
            final_state = {}
            for domain, ds in parsed_state.items():
                if domain in available_domains:
                    final_state[domain] = ds
            
            for domain, dbs in final_state.items():
                if domain not in total_state:
                    total_state[domain] = dbs
                else:
                    for slot, value in dbs.items():
                        value = str(value)
                        if value not in [None, 'dontcare', 'none', '?', '', ' ', "None", "null", "Null", "NULL", "any", "Any"] and len(value) > 0:
                            total_state[domain][slot] = value # update or add slotvalue in total_dict 
            


            if args.print:
                print('-' * 100)
                print(f"Question: {question}", flush=True)
                logger.info(f"Question: {question}")
                print(f"Selected domain: {selected_domain}", flush=True)
                logger.info(f"Raw State: {state}")
                print(f"Raw State: {state}", flush=True)
                logger.info(f"Parsed State: {final_state}")
                print(f"Parsed State: {final_state}", flush=True)
                logger.info(f"Total State: {total_state}")
                print(f"Total State: {total_state}", flush=True)


            # GET DB RESULTS
            database_results = {domain: len(database.query(domain=domain, constraints=ds))
                                for domain, ds in total_state.items() if len(ds) > 0}

            if args.print:
                print(f"Database Results: {database_results}", flush=True)
                logger.info(f"Database Results: {database_results}")
            
            # PREDICT RESPONSE
            try:
                kwargs = {
                    "history": "\n".join(history),
                    "utterance": question.strip(),
                    "state": json.dumps(total_state), #.replace("{", '<').replace("}", '>'),
                    "database": str(database_results)
                }
                # no zero shot for response genration
                kwargs["positive_examples"] = response_examples
                kwargs["negative_examples"] = []
                
                # few shot in all cases
                response, filled_prompt = response_model(response_prompt, predict=True, **kwargs)
                if n < 2 and args.print:
                    print("Filled response prompt:", filled_prompt)
                    logger.info(f"Filled response prompt: {filled_prompt}")
            except:
                response = ''

            response_raw = response
            response = parse_response(response)
            response = delexicalise(response, delex_dic)
            response = delexicaliseReferenceNumber(response)

        if args.print:
            print(f"Response Raw: {response_raw}", flush=True)
            logger.info(f"Response Raw: {response_raw}")
            print(f"Response: {response}", flush=True)
            logger.info(f"Response: {response}")
            print(f"Gold Response: {gold_response}", flush=True)
            logger.info(f"Gold Response: {gold_response}")

        
        # append to history
        history.append("Customer: " + question)
        history.append("Assistant: " + gold_response)

        # add to results
        results[dialogue_id].append({
                "question": question,
                "domain": selected_domain,
                "active_domains": [selected_domain],
                "response": response,
                "state": copy.deepcopy(total_state), 
            })
        results_wo_state[dialogue_id].append({
                "domain": selected_domain,
                "active_domains": [selected_domain],
                "response": response,
            })
            
    progress_bar.close()
    with open(os.path.join(args.results_dir, out_file_name), 'w') as f:
        json.dump(results, f, indent=2)
    
    os.makedirs(os.path.join(args.results_dir, 'eval'), exist_ok=True)

    if args.dataset_variant == 'interference':
        evaluator = MWEvaluator(bleu=True, success=True, richness=True, jga=True, dst=True, filter=True, bleu_aug=True)
    else:
        evaluator = MWEvaluator(bleu=True, success=True, richness=True, jga=True, dst=True, filter=True)
    eval_results = evaluator.evaluate(results)
    with open(os.path.join(args.results_dir, 'eval', out_file_name), 'w') as f:
        json.dump(eval_results, f, indent=2)
   



        

        







