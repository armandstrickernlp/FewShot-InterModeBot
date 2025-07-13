from datasets import load_dataset, load_from_disk
from collections import defaultdict
from typing import Dict, List
from database import MultiWOZDatabase
import json


def load_mwoz(database_path, context_size, split='train', total=10, shuffle=True, available_domains=None, only_single_domain=False, restrict_domains=None, variant=None):
    database = MultiWOZDatabase(database_path)
    # dataset = load_dataset('multi_woz_v22', cache_dir='./data/multi_woz_v22') # needs to be online
    dataset = load_from_disk('./data/hf_mwoz_22/') # download first and then load from disk once on node
    possible_ids = None
    if variant == 'interference' or split in ['valid', 'test']:
        with open(f'data/interference_data/{split}.json') as f:
            interference = json.load(f)
        possible_ids = set(interference.keys()) # only use ids that have interference counterpart for eval
    if variant == 'fused':
        with open(f'data/fusedchat_prepended.json') as f:
            fused = json.load(f)
    if available_domains is not None:
        domain_counts = {d: 0 for d in available_domains}
    else:
        domain_counts = defaultdict(int)
        domain_counts['aux'] = -1
    if shuffle:
        data = dataset[split].shuffle()
    else:
        data = dataset[split]
    n = 1
    slots_per_domain = defaultdict(set)
    domain_counter = defaultdict(int)

    for i, dialog in enumerate(data):
        if possible_ids and dialog['dialogue_id'].split('.')[0] not in possible_ids:
            continue
        if only_single_domain and len(dialog['services']) != 1:
            continue
        if all((dc >= total for dc in domain_counts.values())) or (available_domains is None and n >= total):
            break
        dialogue_id = dialog['dialogue_id'].split('.')[0].lower()
        if len(dialog['services']) > 0:
            domain_gt = dialog['services'][0]
        else:
            domain_gt = ''
        for dom in dialog['services']:
            domain_counter[dom] += 1
        if restrict_domains is not None and not all((dom in restrict_domains for dom in dialog['services'])):
            continue
        if domain_counts[domain_gt] >= total:
            continue
        domain_counts[domain_gt] += 1
        n + 1
        last_state = {}
        
        for tn in range(0, len(dialog['turns']['utterance']), 2):
            # print(tn)
            if variant == 'interference':
                backstory_idx, reac_idx = interference[dialogue_id.upper()]['augmented_idx']
                context, response = from_interfere(dialog, dialogue_id, tn, interference, backstory_idx, reac_idx)
            elif variant == 'fused':
                context = from_fused(dialog, dialogue_id, tn, fused)
                response = delexicalize_mwoz(dialog['turns']['utterance'][tn+1],
                                             dialog['turns']['dialogue_acts'][tn+1]['span_info'])
            else:
                context = [f"Customer: {t}" if n % 2 == 0 else f"Assistant: {t}"
             for n, t in enumerate(dialog['turns']['utterance'][:tn+1])]
                response = delexicalize_mwoz(dialog['turns']['utterance'][tn+1],
                                             dialog['turns']['dialogue_acts'][tn+1]['span_info'])
            
            state = dialog['turns']['frames'][tn]['state']
            if len(state) == 0:
                state = {}
            else:
                state = state[0]['slots_values']
                state = {k: v[0] for k, v in zip(state['slots_values_name'], state['slots_values_list']) }
            new_state = {}
            for sl, val in state.items():
                domain, name = sl.split('-')
                slots_per_domain[domain].add(name)
                if domain not in new_state:
                    new_state[domain] = {name: val}
                else:
                    new_state[domain][name] = val
            state_update = {}
            for domain, domain_state in new_state.items():
                for slot, value in domain_state.items():
                    if slot not in last_state.get(domain, {}) or last_state[domain][slot] != value:
                        if domain not in state_update:
                            state_update[domain] = {}
                        state_update[domain][slot] = value
            last_state = new_state
            database_results = {domain: len(database.query(domain, domain_state))
                                for domain, domain_state in new_state.items()}


            turn = {'page_content': '\n'.join(context[-context_size:]),
                    'question': context[-1].replace("Customer: ", ""),
                    'gt_state': last_state,
                    'dialogue_id': dialogue_id,
                    'metadata': {'domain': f'{domain_gt}',
                                 'state': state_update,
                                 'full_state': last_state,
                                 'context': '\n'.join(context),
                                 'context_list': context,
                                 'response': response,
                                 'database': database_results}}
            # # pp.pprint(context)
            # # pp.pprint(turn['metadata']['context'])
            # print(turn['question'])
            # print(turn['metadata']['response'])
            # print()
            
            yield turn



def from_interfere(dialog, dialogue_id, tn, interference, backstory_idx, reac_idx):
    # return ctxt + response 
    backstory = interference[dialogue_id.upper()]['log'][backstory_idx]['backstory']
    reac = interference[dialogue_id.upper()]['log'][reac_idx]['reaction']
    context = []
    for n, t in enumerate(dialog['turns']['utterance'][:tn+1]):
        if n % 2 == 0:
            custom = f"Customer: {t}"
            if n == backstory_idx:
                custom += f" {backstory}"
            context.append(custom)
        else:
            if n == reac_idx:
                assist = f"Assistant: {reac} {t}"
            else:
                assist = f"Assistant: {t}"
            context.append(assist)
    
    response = delexicalize_mwoz(dialog['turns']['utterance'][tn+1],
                                 dialog['turns']['dialogue_acts'][tn+1]['span_info'])
    if tn+1 == reac_idx:
        response = f"{reac} {response}"
    
    return context, response


def from_fused(dialog, dialogue_id, tn, fused):
    # add prepended turns to context
    fchat_turns = fused[dialogue_id.upper()]["turns"]
    fchat_types = fused[dialogue_id.upper()]["types"]
    context = []
    rewrite = False 
    for idx, (typ, turn) in enumerate(zip(fchat_types, fchat_turns)):
        string_encode = turn.encode("ascii", "ignore") # get rid of unicode characters in fusedchat
        turn = string_encode.decode()
        if typ == 'prepended':
            if idx % 2 == 0:
                utt = f"Customer: {turn}"
            else:
                utt = f"Assistant: {turn}"
            context.append(utt)
        elif typ == "rewritten":
            # only occur on user turns
            context.append(f"Customer: {turn}")
            rewrite = True
            break
        elif typ == 'original':
            break
    for n, t in enumerate(dialog['turns']['utterance'][:tn+1]):
        if rewrite and n == 0:
            continue # already added to context
        if n % 2 == 0:
            custom = f"Customer: {t}"
            context.append(custom)
        else:
            assist = f"Assistant: {t}"
            context.append(assist)
    
    return context


def delexicalize_mwoz(utterance: str, span_info: Dict[str, List[str]]):
    for s_idx in range(len(span_info['act_slot_name']) - 1, -1, -1):
        name = span_info['act_slot_name'][s_idx]
        placeholder = f'[{name}]'
        utterance = utterance[:span_info['span_start'][s_idx]] + placeholder + utterance[span_info['span_end'][s_idx]:]
    return utterance

