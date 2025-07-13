import argparse
import pickle
import json
import logging
import random
import copy
import os
import pprint as pp
from tqdm import tqdm
from collections import Counter

import streamlit as st

import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch


from model import (
    FewShotOpenAIChatLLM,
    ZeroShotOpenAIChatLLM,
    FewShotLlama,
    ZeroShotLlama,
    )

from loaders import load_mwoz
from delex import prepareSlotValuesIndependent, delexicalise, delexicaliseReferenceNumber

from definitions.base import MW_FEW_SHOT_DOMAIN_DEFINITIONS, MW_ZERO_SHOT_DOMAIN_DEFINITIONS, multiwoz_domain_prompt
from definitions.chitchat import chitchat_prompt, task_selection_prompt, domain_selection_prompt
from definitions.function_call import FC_FEW_SHOT_DOMAIN_DEFINITIONS, FC_ZERO_SHOT_DOMAIN_DEFINITIONS
from definitions.sql import SQL_FEW_SHOT_DOMAIN_DEFINITIONS

from database import MultiWOZDatabase
from utils import (parse_state, 
                   parse_func_call,  
                   parse_func_call_gpt,
                   parse_task,
                   parse_sql, 
                   parse_response, 
                   parse_domain, 
                   ExampleRetriever, 
                   ExampleFormatter
                )

set_seed(42)

@st.cache_resource
def init_model():
    if args.model_name.startswith("gpt-"):
        state_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=True, seed=42)
        response_model = FewShotOpenAIChatLLM(args.model_name, seed=42)
        domain_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=False, seed=42)
        chitchat_model = ZeroShotOpenAIChatLLM(args.model_name, func_call=False, seed=42)
        
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
    return state_model, response_model, domain_model, chitchat_model


@st.cache_resource
def init_utils():
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
    return database, state_vs, delex_dic, example_retriever, state_retriever, example_formatter


@st.cache_data
def init_session_state(batch):
    id_list = [idx for idx in range(len(batch))]
    idx2goal = {idx: goal for idx, goal in enumerate(batch)}
    for idx, _ in idx2goal.items():
        idx2goal[idx]["finished"] = False
    st.session_state['id_list'] = id_list
    st.session_state['idx2goal'] = idx2goal
    st.session_state['example_selected'] = False
    st.session_state['results'] = {}



def run_app():

    # introduce task
    if "introduced" not in st.session_state or not st.session_state['introduced']:
        st.title("Welcome to the dialogue completion task!")
        st.markdown("In this task, you will be completing dialogues with an AI assistant from the Cambridge visitor center. Please follow the instructions below.")
        st.markdown("- For each dialogue you will be given a **task goal**, which includes a **seed turn** to copy paste and start the conversation and **next steps** to follow.")
        st.markdown("- Possible domains for the tasks are: **train**, **restaurant**, **hotel**, **attraction**, **taxi**.")
        st.markdown("- Your task is to interact with the assistant to **complete the task**")
        st.markdown("- After each dialogue, you will be asked to rate the assistant's **friendliness**, **task completion**, and the number of **clarifications**, if any were needed.")

        def intro():
            st.session_state["introduced"] = True
        
        st.button("Start Task", on_click=intro)
        return


    # run sidebar
    st.sidebar.title('Task List')
    st.sidebar.markdown("Please select a goal.")
    button_placeholders = [st.sidebar.empty() for i in range(len(st.session_state['id_list']))]

    def select_example(idx):
        st.session_state['current_selection'] = st.session_state['idx2goal'][idx]
        st.session_state['selected_index'] = idx
        st.session_state['example_selected'] = True

    for idx in st.session_state['id_list']:
        if idx == 0:
            button_placeholders[idx].button(f"PRACTICE", key=f"selection_{idx}", on_click=select_example, args=(idx,), use_container_width=True)
        else:
            button_placeholders[idx].button(f"Dialogue {idx}", key=f"selection_{idx}", on_click=select_example, args=(idx,), use_container_width=True)
        
        if st.session_state['idx2goal'][idx]['finished']:
            button_placeholders[idx].button(f"Dialogue {idx} (finished)", key=f"selection_finished_{idx}", on_click=select_example, args=(idx,), use_container_width=True, disabled=True)
    
    # sidebar form (final submit button)
    with st.sidebar.form(key='my-form'):
        warning_placeholder = st.empty()
        final_submit = st.form_submit_button('Submit', type='secondary', use_container_width=True)
    
    # if final submit button is clicked, check if all dialogues are finished and name is entered
    if final_submit:
        for idx in st.session_state['id_list']:
            if not st.session_state['idx2goal'][idx]['finished']:
                warning_placeholder.warning("Please finish all dialogues before submitting.")
                return
            
        # save results
        with open(f"{args.results_dir}/results_{args.batch_num}_{args.usr_name}.json", 'w') as f:
            json.dump(st.session_state["results"], f, indent=2)
                        
        st.title("Thank you for participating!")
        st.balloons()
        st.stop()
    

    # dialogue page
    if st.session_state['example_selected']:

        REFRESH = False
        # database_results_cnt = None

        # Initialize chat history
        if "messages" not in st.session_state["current_selection"]:
            st.session_state["current_selection"]["messages"] = []
        
        if not REFRESH: 
            if st.session_state["current_selection"]["messages"] and not st.session_state["current_selection"]["finished"]:
                for message in st.session_state["current_selection"]["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content_raw"])
        else:
            st.session_state["current_selection"]["messages"] = []
        
        # init total_state
        if "total_state" not in st.session_state["current_selection"]:
            st.session_state["current_selection"]["total_state"] = {}
        total_state = st.session_state["current_selection"]["total_state"]

        # init eval state
        if "QA" not in st.session_state["current_selection"]:
            st.session_state["current_selection"]["QA"] = {}
            st.session_state["current_selection"]["QA"]["chitchat"] = ""
            st.session_state["current_selection"]["QA"]["success"] = ""
            st.session_state["current_selection"]["QA"]["corrections"] = ""


        if usr_input := st.chat_input(""):
    
            st.session_state["current_selection"]["messages"].append({"role": "customer", 
                                                                      "content_raw": usr_input,
                                                                      "content": f'Customer: {usr_input}'})
            with st.chat_message("customer"):
                st.markdown(usr_input)


            with st.chat_message("assistant"):
                
                history = [msg['content'] for msg in st.session_state["current_selection"]["messages"]]
                print(f"HISTORY: {history}")

                # predict turn type (chitchat or task)
                if len(history) < 5:
                    prev = "\n".join(history[-3:-1])
                else:
                    prev = "\n".join(history[-2:-1])
                task, dp = domain_model(task_selection_prompt, predict=True, history=prev, utterance=f"Customer: {usr_input.strip()}")
                print("TASK PROMPT: ", dp)
                selected_task = parse_task(task)
                print(f"TASK: {selected_task}")
                
            
                if selected_task == 'chitchat':
                    response, filled_prompt = chitchat_model(chitchat_prompt, predict=True, history="\n".join(history[:-1]), utterance=f"Customer: {usr_input.strip()}")
                    response = parse_response(response)
                    st.markdown(response)
                    st.session_state["current_selection"]["messages"].append({"role": "assistant", 
                                                                              "content_raw": response,
                                                                              "content": f'Assistant: {response}', 
                                                                              "current_state": copy.deepcopy(total_state)})
                                                                             # no update to state
                else: 
                    # get domain
                    domain, dp = domain_model(domain_selection_prompt, predict=True, history="\n".join(history[-3:-1]), utterance=f"Customer: {usr_input.strip()}")
                    selected_domain = parse_domain(domain)

                    print(f"SELECTED DOMAIN: {selected_domain}")

                    available_domains = [dom for dom in MW_FEW_SHOT_DOMAIN_DEFINITIONS.keys() if dom != 'hospital']
                    if selected_domain not in available_domains: # pick random if not in possible choices
                        selected_domain = random.choice(available_domains)
                    # load prompt class for predicted domain

                    if args.model_name.startswith("meta-llama") :
                        domain_definition = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain]
                        state_prompt = domain_definition.state_prompt
                        response_prompt = domain_definition.response_prompt
                        expected_slots = domain_definition.expected_slots
                    
                    elif args.model_name.startswith("gpt-"):
                        state_prompt = FC_ZERO_SHOT_DOMAIN_DEFINITIONS[selected_domain].state_prompt
                        response_prompt = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].response_prompt # classic few shot response prompt
                        expected_slots = FC_FEW_SHOT_DOMAIN_DEFINITIONS[selected_domain].expected_slots # few or zero doesnt matter
                    
                    # make state and response prompts
                    retrieved_examples = example_retriever.retrieve("\n".join(history[-args.context_size:]), k=5)
                    retrieved_examples = [example for example in retrieved_examples]
                    num_examples = min(len(retrieved_examples), args.num_examples)

                    num_state_examples = args.num_examples
                    state_examples = [example for example in state_retriever.retrieve("\n".join(history[-args.context_size:]), k=20)][:num_state_examples]
                    
                    positive_state_examples = example_formatter.format(state_examples[:num_state_examples],
                                                                input_keys=["context"],
                                                                output_keys=["state"],
                                                                func_call=True,
                                                                )
                    response_examples = example_formatter.format(retrieved_examples[:num_examples],
                                                            input_keys=["context", "full_state", "database"],
                                                            output_keys=["response"],
                                                            use_json=True)
                    
                    # PREDICT STATE
                    try:
                        kwargs = {
                            "history": '\n'.join(history[:-1]),
                            "utterance": usr_input.strip()
                        }
                        if args.model_name.startswith("meta-llama"):
                            kwargs["positive_examples"] = positive_state_examples
                            kwargs["negative_examples"] = [] # negative_state_examples not added
                        
                        # PREDICTION
                        state, filled_state_prompt = state_model(state_prompt, predict=True, **kwargs) # PREDICTION
                    except:
                        state = "{}"

                    pp.pprint(f"Raw STATE: {state}")

                    # PARSE STATE
                    if args.model_name.startswith("meta-llama"):
                        parsed_state = parse_func_call(state, default_domain=selected_domain)
                    else:
                        parsed_state = parse_func_call_gpt(state, default_domain=selected_domain)

                    pp.pprint(f"PARSED STATE: {parsed_state}")

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
                    
                    state_copy = copy.deepcopy(parsed_state)

                    for domain, ds in state_copy.items():
                        if domain in ['restaurant', 'hotel', 'attraction']:
                            for slot, value in ds.items():
                                if slot in ['pricerange', 'area', 'food', 'internet', 'parking', 'stars', 'type'] and domain in total_state and total_state[domain].get(slot, None) is not None: 
                                    if total_state[domain][slot] != value: # if the contraint has changed from previous turn, make sure to delete any entity names, so db can be searched with new constraints
                                        if total_state[domain].get('name', None) is not None:
                                            del total_state[domain]['name']
                                        if parsed_state[domain].get('name', None) is not None: 
                                            del parsed_state[domain]['name']
                        if domain in ['train', 'hotel', 'restaurant']:
                            if domain not in total_state:
                                total_state[domain] = {}
                            if total_state[domain].get('bookpeople', None) is None:
                                total_state[domain]['bookpeople'] = '1' # set default booking people to 1
                
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
                    

                    pp.pprint(f"TOTAL STATE: {total_state}")

                    # if selected_domain == 'restaurant':
                    #     if database_results_cnt and database_results_cnt[selected_domain] == 0:
                    #         del total_state[selected_domain]['name']
                    # print(total_state)

                    # GET DB RESULTS
                    database_results = {domain: database.query(domain=domain, constraints=ds)
                                        for domain, ds in total_state.items() if len(ds) > 0}
                    database_results_cnt = {domain: len(results) for domain, results in database_results.items()}
                    pp.pprint(f"DATABASE RESULTS COUNT: {database_results_cnt}")
                    # pp.pprint(f"DATABASE RESULTS: {database_results}")
                    


                    # print([msg['content'] for msg in st.session_state["current_selection"]["messages"]])
                    # if database_results_cnt[selected_domain] == 0 and total_state.get('name', None) is not None:
                    #     for i, turn in enumerate(st.session_state["current_selection"]["messages"]):
                    #         if total_state['name'] in turn['current_state']['content']:
                    #             st.session_state["current_selection"]["messages"][i]["content"].replace(total_state['name'], '[name]')
                    #     del total_state['name']
                    # print([msg['content'] for msg in st.session_state["current_selection"]["messages"]])
                        



                    # PREDICT RESPONSE
                    try:
                        kwargs = {
                            "history": "\n".join(history),
                            "utterance": usr_input.strip(),
                            "state": json.dumps(total_state), #.replace("{", '<').replace("}", '>'),
                            "database": str(database_results_cnt)
                        }
                        # always few shot for response genration
                        kwargs["positive_examples"] = response_examples
                        kwargs["negative_examples"] = []
                        
                        # few shot in all cases
                        response, filled_prompt = response_model(response_prompt, predict=True, **kwargs)

                    except:
                        response = ''
                    
                    # print(f"RESPONSE PROMPT: {filled_prompt}")

                    response_raw = response
                    response = parse_response(response)
                    response_delex = delexicalise(response, delex_dic)
                    response_delex = delexicaliseReferenceNumber(response)



                    def lexicalize(results, domain, response, ds):
                        
                        if domain not in results:
                            return response
                        elif len(results[domain]) == 0:
                            return response
                        item = results[domain][0] # pick first item always

                        # extend dict with possible placeholder anmes
                        extend_dct = {f"{domain}_{key}": val for key, val in item.items()}
                        item.update(extend_dct)
                        item.update({f"value_{key}": val for key, val in item.items()})
                        item.update({'reference': item['ref'] if 'ref' in item else ''})

                        item["choice"] = str(len(results[domain]))

                        # extend with values in state
                        item.update({f"{key}": val for key, val in ds[domain].items()})
                        item.update({f"value_{key}": val for key, val in ds[domain].items()})
                        item.update({f"{domain}_{key}": val for key, val in ds[domain].items()})

                        # match key in response with key in item
                        for key, val in item.items():
                            x = f"[{key}]"
                            if x in response:
                                response = response.replace(x, val)
                        
                        # hard code ref if not replaced
                        response = response.replace('[ref]', '78946').replace('[reference]', '78946')
                        return response
                    
                    response_lex = lexicalize(database_results, selected_domain, response, total_state) # use raw response instead of delex. Model can delex db values well.
                    



                    
                    pp.pprint(f"RAW RESPONSE: {response_raw}")
                    pp.pprint(f"RESPONSE DELEX: {response_delex}")
                    pp.pprint(f"RESPONSE: {response_lex}")

                    st.session_state["current_selection"]["messages"].append({"role": "assistant", 
                                                                            "content_raw": response_lex,
                                                                            "content": f'Assistant: {response_lex}',
                                                                            "current_state": copy.deepcopy(total_state), 
                                                                            "db_results_cnt": database_results_cnt,
                                                                            "db_result": copy.deepcopy(database_results)})
                    st.markdown(response_lex)
        

        
    
        # tabs
        tab1, tab2 = st.tabs(["Task Goal", "QA"])
        with tab1:
            st.markdown(f"**Dialogue {st.session_state['selected_index']}**", unsafe_allow_html=True)
            st.markdown(f"START TURN:  \n{st.session_state['current_selection']['seed_turn']}", unsafe_allow_html=True)
            st.markdown(f"NEXT STEPS: \n{st.session_state['current_selection']['steps']}", unsafe_allow_html=True)
        
        with tab2:
            chitchat = st.text_area("When chatting, was the assistant friendly and engaging? Pick a number from **1** (worst) to **5** (best)", value=st.session_state["current_selection"]["QA"]["chitchat"])
            success = st.text_area("Was the assistant able to complete the task, giving you the information you needed? (Y/N)", value=st.session_state["current_selection"]["QA"]["success"])
            corrections = st.text_area("How many corrections/clarifications were needed?", value=st.session_state["current_selection"]["QA"]["corrections"])
            
            st.session_state["current_selection"]["QA"]["chitchat"] = chitchat
            st.session_state["current_selection"]["QA"]["success"] = success
            st.session_state["current_selection"]["QA"]["corrections"] = corrections

            def finish_dialogue():
                if not st.session_state["current_selection"]["QA"]["chitchat"] or not st.session_state["current_selection"]["QA"]["success"] or not st.session_state["current_selection"]["QA"]["corrections"] :
                    st.warning("Please fill out QA fields before finishing.")
                    return
                st.session_state["current_selection"]["finished"] = True
                pp.pprint(st.session_state["current_selection"])
                st.session_state["results"][st.session_state['selected_index']] = st.session_state["current_selection"]
                with open(f"{args.results_dir}/results_{args.batch_num}_{args.usr_name}.json", 'w') as f:
                    json.dump(st.session_state["results"], f, indent=2)
            
            
            st.button("Finish Dialogue", on_click=finish_dialogue)
        
            

                






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument("--batch_num", type=int, default=0, help="Batch number.")
    parser.add_argument("--batches_path", type=str, default='interact/data/batches.json')
    parser.add_argument("--usr_name", type=str, default='armand')
    

    parser.add_argument("--results_dir", type=str, default='interact/results/gpt-turbo')
    parser.add_argument("--cache_dir", type=str, default='./model_cache')

    parser.add_argument("--database_path", type=str, default="multiwoz_database")
    parser.add_argument("--faiss_db", type=str, default="multiwoz-context-db.vec")
    parser.add_argument("--ontology", type=str, default="ontology.json")

    parser.add_argument("--context_size", type=int, default=3) # number of prev turns to consider for retrieval
    parser.add_argument("--num_examples", type=int, default=5) # number of 'shots'
    args = parser.parse_args()

    # print(args)

    # init
    os.makedirs(args.results_dir, exist_ok=True)
    state_model, response_model, domain_model, chitchat_model = init_model()
    database, state_vs, delex_dic, example_retriever, state_retriever, example_formatter = init_utils()
   

    # load goal batches and init session state
    with open(args.batches_path) as f:
        goal_batches = json.load(f)
    selected = goal_batches[args.batch_num]
    init_session_state(selected)

    # run app
    run_app()






    




    



