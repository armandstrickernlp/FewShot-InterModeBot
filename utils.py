import json
import re
import dirtyjson
import random
from pynvml import *
from copy import deepcopy
from typing import Dict, Any, Text
from collections import defaultdict

import numpy
from fuzzywuzzy import fuzz

from nltk.tokenize import word_tokenize
from langchain_community.vectorstores import VectorStore

from definitions.function_call import DOMAIN_2_FUNCTION, FUNCTION_2_DOMAIN

import pprint as pp

def parse_state(state: str, default_domain: str = None):
    def sanitize(dct):
        for key in dct:
            if isinstance(dct[key], dict):
                dct[key] = sanitize(dct[key])
            elif not isinstance(dct[key], str):
                dct[key] = str(dct[key])
        return dct

    state = str(state)
    state = state.replace("state: ", "").strip() # remove state: prefix if model outputs it
    slotvals = re.findall("([a-z]+: ?('(([a-z]| |[A-Z]|:|[0-9])+')|[A-Za-z0-9: ]+))", state) # remove single quote from [a-z]+ and add space in npossible characters
    # slotvals = re.findall("([a-z]+:('(([a-z]| |[A-Z]|:|[0-9])+')|[A-Za-z0-9:]+))", state)
    out_state = {}
    for sv in slotvals:
        sv = sv[0].strip("'\"").split(':')
        out_state[sv[0].strip("'\"")] = ":".join(sv[1:]).strip("'\" ")
    return {default_domain: sanitize(out_state)} # added default_domain nesting

def parse_func_call(output, default_domain=None):
    output = str(output)
    pattern = r"(?:<<function>>)?(\w+)\(\{?(.*?)\}?\)"
    match = re.search(pattern, output)
    if not match:
        return {default_domain: {}}

    function_name = match.group(1)
    arguments_str = match.group(2).strip()
    arguments_str = arguments_str.replace(": ", "=")
    arguments_parts = arguments_str.split(", ")
    arguments = {}
    for part in arguments_parts:
        try:
            key, value = part.split("=")
            key = key.strip().replace("'", "").replace('"', "")
            value = value.strip().replace("'", "").replace('"', "")
            if value not in ["", " ", None, "None", "none", "null", "Null", "NULL", "any", "Any", '<missing>']:
                arguments[str(key.strip())] = str(value.strip()) # make sure key and value are strings just in case
        except ValueError:
            print(f"An error occurred while parsing: {part}")
            print(f"Arguments: {arguments_str}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            continue
    domain = FUNCTION_2_DOMAIN.get(function_name, None)
    out_state = {domain: arguments}
    return out_state

def parse_func_call_gpt(output, default_domain=None):
    # output should be a tuple (func_name, args)
    if isinstance(output, tuple):
        try:
            fc_name, args = output[0], output[1]
            args = args.replace('None', 'null') # dirtyjson does not deal with None
            args = dirtyjson.loads(args)
        except Exception as e:
            return {default_domain: {}}
    else:
        return {default_domain: {}}

    if isinstance(args, str):
        args = dirtyjson.loads(args)
    domain = FUNCTION_2_DOMAIN.get(fc_name, None)
    arguments = {}
    for key, val in args.items():
        key, val = str(key).strip(), str(val).strip() 
        if key == 'area' and val not in ['west', 'east', 'north', 'south', 'centre']:
            continue
        elif val not in ["", " ", None, "None", "none", "null", "Null", "NULL", "any", "Any", '<missing>']:
            arguments[key] = val
        elif val not in ['not specified', 'not provided', 'undefined', 'not given', 'not available', 'unknown', 'no', 'nope', 'n/a', 'none', 'null', 'nil', 'missing']:
            arguments[key] = val

    out_state = {domain: arguments}
    return out_state

def parse_sql(query, default_domain=None):
    pattern = r"SELECT\s+.*?\s+FROM\s+([\w\s]+)\s+WHERE\s+(.*?);"
    match = re.search(pattern, query, re.IGNORECASE | re.DOTALL)
    if not match:
        return {default_domain: {}}

    domain = match.group(1).strip()
    conditions_str = match.group(2)

    condition_pattern = r"(\w+)\s*(=|>|<|>=|<=)\s*('(?:[^']*)'|\b\S+\b)"
    matches = re.finditer(condition_pattern, conditions_str)
    if not matches:
        return {domain: {}}
    arguments = {}
    for match in matches:
        key = match.group(1).strip().replace("'", "").replace('"', "")
        value = match.group(3).strip().replace("'", "").replace('"', "")
        if value not in ["", " ", "None", "none", "null", "Null", "NULL", "any", "Any", '<missing>']:
            arguments[key] = value

    return {domain: arguments}

def parse_response(response):
    pattern = r"<response>(.*?)</response>"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()
    else:
        pattern = r"(?:Response:|Assistant:|</response>|<response>)\s*(.*?)(?=\n|$)"
        match = re.search(pattern, response, re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()

def parse_domain(output):
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
    pattern = r'(?:' + '|'.join(domains) + ')'
    match = re.search(pattern, output, re.IGNORECASE) 
    return match.group(0).lower() if match else ''

def parse_task(output):
    tasks = ['task', 'chitchat']
    pattern = r'(?:' + '|'.join(tasks) + ')'
    match = re.search(pattern, output, re.IGNORECASE)
    return match.group(0).lower() if match else ''

class ExampleRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, text: str, k: int = 2) -> list[Dict]:
        result = self.vector_store.similarity_search(text, k=k)
        examples = [{'context': doc.metadata['context'],
                     'state': doc.metadata['state'],
                     'full_state': doc.metadata['full_state'],
                     'response': doc.metadata['response'],
                     'database': doc.metadata['database'],
                     'domain': doc.metadata['domain']}
                     for doc in result]
        return examples
    


class ExampleFormatter:
    def __init__(self, ontology: Dict):
        self.ontology = ontology
        self.func_names = DOMAIN_2_FUNCTION


    def format(self,
               examples: list[Dict[str, Any]],
               input_keys: list[str],
               output_keys: list[str],
               use_json: bool = False,
               corrupt_state: bool = False,
               func_call: bool = False,
               write_sql: bool = False,
               ) -> list[Dict[str, str]]:

        examples = deepcopy(examples)
        if corrupt_state:
            examples = [self._corrupt_example(example) for example in examples]
        
        
        for example in examples:
            state_domains = list(example['state'].keys())
            if len(state_domains) > 0:
                example['state'] = example['state'][state_domains[0]] # flatten the state (take val of first domain)
            else:
                example['state'] = {}

        examples = [self._example_to_str(example, use_json, func_call, write_sql) for example in examples]
        # pp.pprint(examples)

        def _prepare_example(example: Dict) -> Dict:
            example['input'] = '\n'.join((f"{key if key != 'full_state' else 'state'}:{example[key]}" for key in input_keys))
            example['output'] = '\n'.join((f"{key}: {example[key]}" if key not in ['state', 'response'] else f"Assistant: {example[key]}" for key in output_keys))
            return example
        examples = [_prepare_example(example) for example in examples]
        # pp.pprint(examples)

        return examples
    
    
    def _example_to_str(self, example: Dict, use_json=False, func_call=False, write_sql=False) -> Dict:
        for key, val in example.items():
            if isinstance(val, dict):
                if use_json:
                    example[key] = json.dumps(val) # .replace("{", '<').replace("}", '>')
                
                if key == 'state' and func_call:
                    func_name = self.func_names[example['domain']]
                    params = ', '.join([f"{type}={value}" for type, value in val.items()])
                    example[key] = f"""<<function>>{func_name}({params})"""
                
                elif key == 'state' and write_sql:
                    arguments = ' AND '.join([f"{type} = {value}"for type, value in val.items()])
                    example[key] = f"SELECT * FROM {example['domain']} WHERE {arguments};"
                
                else:
                    example[key] = "-".join((f"{slot}:'{value}'" for slot, value in val.items()))
            elif key =='response':
                example[key] = f"<response> {val} </response>"
            else:
                example[key] = str(val)
        return example
    
    def _corrupt_example(self, example: Dict) -> Dict:
        for domain, dbs in example['state'].items():
            for slot, value in dbs.items():
                slot_otgy_name = f"{domain}-{slot}"
                if slot_otgy_name in self.ontology:
                    example['state'][domain][slot] = random.choice(self.ontology[slot_otgy_name])
                else:
                    otgy_key = random.choice(list(self.ontology.keys()))
                    example['state'][domain][slot] = random.choice(self.ontology[otgy_key])
        return example


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


# def parse_func_call(call, default_domain: str = None):
#     def clean(params):
#         new_params = {}
#         for k, v in params.items():
#             if v not in [None, '', "None", "none", "null", "Null", "NULL", "any", "Any"]:
#                 new_params[k] = str(v)
#         return new_params
    
#     start_tag, end_tag = "<function_call>", "</function_call>"
#     start_idx = call.find(start_tag) + len(start_tag)
#     end_idx = call.find(end_tag)
#     try:
#         json_str = call[start_idx:end_idx].strip()
#         data = json.loads(json_str)
#         domain = FUNCTION_2_DOMAIN[data["function_name"]]
#         params = clean(data["parameters"])
#         out_state = {domain: params}
    
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#         out_state = {default_domain: {}}
        
#     return out_state