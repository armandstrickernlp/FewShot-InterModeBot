from db_ops import MultiWozDB
import json
import os

from normalize_slot_values import normalize_state_slot_value
# notes on normalizing
# leave dataset bs values without normalization as these often correspond to what's in the dialogue context and makes sense to want the model to learn to pick these out from the context
# add normalization when searching db though, to improve likeliness of finding a match

class MwozDatasetConstructor:
    def __init__(self, multiwoz, fusedchat, interference, variant='fused'):
        # variant can be 'mwoz', 'fused' or 'interfere'
        # context applies to fusedchat
        # interferences are merged with original turn
        self.multiwoz = multiwoz
        self.fusedchat = fusedchat
        self.interference = interference
        self.variant = variant
        self.db = MultiWozDB(db_path='./db')
        self.db_state_tokens = [
            ['[db_state0]','[db_state1]','[db_state2]','[db_state3]','[db_state4]'], 
            ['[db_state0+bookfail]', '[db_state1+bookfail]','[db_state2+bookfail]','[db_state3+bookfail]','[db_state4+bookfail]'],
            ['[db_state0+booksuccess]','[db_state1+booksuccess]','[db_state2+booksuccess]','[db_state3+booksuccess]','[db_state4+booksuccess]']
            ]
        
    def delexicalize(self, turn):
        text_delex = turn["text"]
        delex_spans = turn["span_info"]
        char_diff = 0
        for span in delex_spans:
            act, slot, value, start, end = span
            start += char_diff
            end += char_diff
            len1 = len(text_delex)
            text_delex = text_delex[:start] + f'[{slot}]' + text_delex[end:]
            len2 = len(text_delex)
            char_diff += len2 - len1
        return text_delex
    
    def get_state_from_metadata(self, metadata):
        gold_dict = {}
        for domain in metadata:
            if domain == 'bus': # IGNORE bus, not in db
                continue
            domain_dict = {}
            constraint = metadata[domain]
            for slot in constraint['book']:
                if slot != 'booked' and constraint['book'][slot]:
                    if domain == 'train' and slot == 'ticket':
                        domain_dict['people'] = constraint['book'][slot][0] # replace ticket slot by people
                    else:
                        domain_dict[slot.lower()] = '|'.join(constraint['book'][slot]) # add list, multiple values may be possible
                                
            for slot in constraint['semi']:
                if  constraint['semi'][slot]: 
                    domain_dict[slot.lower()] = '|'.join(constraint['semi'][slot])

            if domain_dict:
                gold_dict[domain] = domain_dict 
        return gold_dict
    
    def update_seen(self, seen_domains, current_state):
        for domain in list(current_state.keys()):
            if domain not in seen_domains:
                seen_domains.append(domain)
        # handle annotation errors where next bs_state domains dont match previous
        for domain in seen_domains:
            if domain not in current_state and domain != 'general':
                print(f"Annotation error: deleting domain {domain} from seen_domains because not in current_state {current_state}")
                seen_domains.remove(domain)    
        return seen_domains        
            

    def get_actions(self, turn_acts):
        action = '<action> '
        name_acts = [] 
        other_acts = []
        for act in turn_acts:        
            for slot, _ in turn_acts[act]:
                act = act.replace('-', ' ').lower() # Hotel-Inform => hotel inform
                if slot.lower() == "name":
                    name_acts.append(f"{act} {slot.lower()}")
                elif slot == 'none':
                            other_acts.append(act)
                else:
                    other_acts.append(f"{act} {slot.lower()}")
        list_acts = name_acts + other_acts
        action += ', '.join(list_acts)
        action += ' </action>'
        return action

    def flatten_state(self, state):
        flattened_string = ''
        for domain in state:
            for slot in state[domain]:
                flattened_string += f"{domain} {slot} {state[domain][slot].split('|')[0]}, " # pick first slot value for training
        if flattened_string:
            flattened_string = flattened_string[:-2]
        flattened_string = f'<belief> {flattened_string} </belief>'
        return flattened_string

    def get_DB_token(self, constraint_dict, dial_act, current_domain):
        if current_domain != 'general':
            try: 
                constraint_dict[current_domain]
            except KeyError:
                print(constraint_dict)
                print(current_domain)   
            for slot in constraint_dict[current_domain]:
                # pick first option if multiple values
                # and normalize to improve likeliness of finding a match in the db
                constraint_dict[current_domain][slot] = normalize_state_slot_value(slot, constraint_dict[current_domain][slot].split('|')[0])
        
        matnums = self.db.get_match_num(constraint_dict)
        match = matnums[current_domain]
        dbvec = self.db.addDBPointer(current_domain, match)
        try:
            db_state = dbvec.index(1) +1
        except ValueError:
            db_state = 0

        bkvec = self.db.addBookingPointer(dial_act)
        try:
            bk_state = bkvec.index(1) +1
        except ValueError:
            bk_state = 0
        return self.db_state_tokens[bk_state][db_state], match
    
    def make_input_output_sys_msg(self, context, bs_state, db_state, actions, response):
        fields = {}
        fields['input'] = f"<context> {context}</context> "
        fields['output'] = f"{bs_state} {db_state} {actions} {response}"
        fields['sys_msg'] = """You are a friendly task-oriented AI assistant. Given a dialogue context, which includes previous dialogue turns, \
you have to generate:
- the belief state which comprises of the user's constraints (time of booking, location, etc.)
- the database state which tells you how many matches there are in the database
- the actions which are the dialogue acts
- the Assistant's response
"""
        return fields

    
    def prep(self, num_list):
        example_dict = {}
        for idx, dial_num in enumerate(num_list): # num_list has to correspond to the dataset_split
            # if idx == 3:
            #     break
            mwoz_dial = self.multiwoz[dial_num+'.json']
            dial_examples = []
            context = ''
            user_request = ''
            seen_domains = ['general']

            if self.variant == 'fused':
                rewrite = False
                fchat_types = self.fusedchat[dial_num]['types']
                fchat_turns = self.fusedchat[dial_num]['turns']
                for idx, (typ, turn) in enumerate(zip(fchat_types, fchat_turns)):
                    string_encode = turn.encode("ascii", "ignore") # get rid of unicode characters in fusedchat
                    turn = string_encode.decode()
                    if typ == 'prepended':
                        if idx % 2 == 0:
                            context += f"<user> {turn} "
                        else:
                            context += f"<system> {turn} "
                    elif typ == "rewritten":
                        # only occur on user turns
                        user_request = ' '.join(turn.split())
                        context += f"<user> {user_request} "
                        rewrite = True
                        break
            
            for idx, turn in enumerate(mwoz_dial['log']):
                if self.variant == 'fused':
                    if rewrite and idx == 0:
                        continue

                if idx % 2 == 0:
                    user_request = ' '.join(turn['text'].split())
                    if self.variant == 'interfere':
                        backstory_idx, _ = self.interference[dial_num]['augmented_idx']
                        if idx == backstory_idx:
                            backstory = self.interference[dial_num]['log'][backstory_idx]['backstory']
                            user_request += f" {backstory}"
                    context += f"<user> {user_request} "

                else:   
                    # Belief state
                    current_state = self.get_state_from_metadata(turn['metadata'])
                    flat_state = self.flatten_state(current_state)
                    seen_domains = self.update_seen(seen_domains, current_state)
                    current_domain = seen_domains[-1] # most recently added domain
                        
                    # DB results
                    db_token, match_nums = self.get_DB_token(current_state, turn['dialog_act'], current_domain)
                    db_state = f"<db> {db_token} </db>"

                    # Action
                    action = self.get_actions(turn['dialog_act'])

                    # Response
                    delex_resp = self.delexicalize(turn)
                    lex_resp = ' '.join(turn['text'].split())
                    if self.variant == 'interfere':
                        _, reac_idx = self.interference[dial_num]['augmented_idx']
                        if idx == reac_idx:
                            reaction = self.interference[dial_num]['log'][reac_idx]['reaction']
                            delex_resp = f"{reaction} {self.delexicalize(turn)}"
                            lex_resp = f"{reaction} {lex_resp}"
                    
                    delex_sys_resp = f"<response> {delex_resp} </response>"

                    # Get booking api
                    if "Booking-NoBook" in turn['dialog_act']:
                        booking_api = 'Fail'
                    elif "Booking-Book" in turn['dialog_act'] or 'Train-OfferBooked' in turn['dialog_act']:
                        booking_api = 'Success'
                    else:
                        booking_api = 'None'
                    


                    eval_data = {
                        'domain': current_domain,
                        'gold_state': current_state,
                        'flat_belief': flat_state,
                        'action': action,
                        'booking_api': booking_api,
                        'db_token': db_token,
                        'current_domain': current_domain,
                        'context': context,
                        'user_request': user_request,
                        'delex_response': delex_sys_resp,
                        'lex_response': lex_resp,
                    }

                    eval_data.update(self.make_input_output_sys_msg(context, flat_state, db_state, action, delex_sys_resp))
                    
                    if self.variant == 'fused' and rewrite and idx == 1:
                        eval_data['rewrite'] = True
                        
                    dial_examples.append(eval_data)
                    context += f"<system> {lex_resp} " 
                    
                example_dict[dial_num] = dial_examples    

        return example_dict   
    
    def prep_lm_data(self, example_dict):
        lm_data = []
        for dial_num in example_dict:
            for turn in example_dict[dial_num]:
                lm_data.append({'input': turn['input'], 
                                'output': turn['output'],
                                'sys_msg': turn['sys_msg']})
        return lm_data   


if __name__ == '__main__':
    data_dir = '../../data/'

    with open(data_dir+'fusedchat_prepended.json') as f:
        fusedchat = json.load(f)

    with open(data_dir+'MultiWOZ_2.2.json') as f:
        multiwoz = json.load(f)

    # interference
    with open(data_dir+'interference_data/train.json') as f:
        train_inter = json.load(f)
    with open(data_dir+'interference_data/valid.json') as f:
        valid_inter = json.load(f)
    with open(data_dir+'interference_data/test.json') as f:
        test_inter = json.load(f)
    dial_nums_train = list(train_inter.keys())
    dial_nums_valid = list(valid_inter.keys())
    dial_nums_test = list(test_inter.keys())
    # combine interference jsons
    interference = {**train_inter, **valid_inter, **test_inter}
    # fused
    constructor = MwozDatasetConstructor(multiwoz, fusedchat=None, interference=None, variant='mwoz')
    # train = constructor.prep(dial_nums_train)
    valid = constructor.prep(dial_nums_valid)
    # test = constructor.prep(dial_nums_test)

    eval_data_dir = 'eval_data'
    lm_data_dir = 'lm_data'
    os.makedirs(eval_data_dir, exist_ok=True)

    for variant in ['mwoz', 'fused', 'interfere']:
        for num_list, split in zip([dial_nums_train, dial_nums_valid, dial_nums_test], ['train', 'val', 'test']):
            constructor = MwozDatasetConstructor(multiwoz, fusedchat, interference, variant=variant)
            example_dict = constructor.prep(num_list)
            lm_example_dict = constructor.prep_lm_data(example_dict)

            os.makedirs(os.path.join(eval_data_dir, variant), exist_ok=True)
            with open(os.path.join(eval_data_dir, variant, f'{split}.json'),'w') as f:
                json.dump(example_dict, f, indent=3)
            
            os.makedirs(os.path.join(lm_data_dir, variant), exist_ok=True)
            with open(os.path.join(lm_data_dir, variant, f'{split}.json'),'w') as f:
                json.dump(lm_example_dict, f, indent=3)
