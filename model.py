from typing import Any, Dict
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import time

load_dotenv('./.env')
client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

from transformers import GenerationConfig
import torch

from prompts import FewShotPrompt, SimpleTemplatePrompt



class SimplePromptedLLM:
    # function call and sys prompt can be None
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
       
    def __call__(self, prompt: SimpleTemplatePrompt, predict=True, **kwargs: Any):
        filled_prompt, system_prompt, function_def, sql_table = prompt(**kwargs) # system_prompt, funciton_call, sql_table can be None cf. SimpleTemplatePrompt
        prediction = self._predict(filled_prompt, system_prompt, function_def, sql_table, **kwargs) if predict else None
        return prediction, filled_prompt

    def _predict(self, text, **kwargs):
        pass
        # input_ids = self.tokenizer.encode(text,return_tensors="pt").to(self.model.device)
        # max_length = max_new_tokens = 50
        # if self.type == 'causal':
        #     max_length = input_ids.shape[1] + max_length
        # output = self.model.generate(input_ids,
        #                              do_sample=False,
        #                              pad_token_id=self.tokenizer.pad_token_id,
        #                              max_new_tokens=max_new_tokens,)
        # if self.type == 'causal':
        #     output = output[0, input_ids.shape[1]:]
        # else:
        #     output = output[0]
        # output = self.tokenizer.decode(output, skip_special_tokens=True)
        # return output


class FewShotPromptedLLM(SimplePromptedLLM):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def __call__(self, prompt: FewShotPrompt, positive_examples: list[Dict], negative_examples: list[Dict], predict=True, **kwargs: Any):
        filled_prompt, system_prompt, function_def, sql_table = prompt(positive_examples, negative_examples, **kwargs)
        prediction = self._predict(filled_prompt, system_prompt, function_def, sql_table, **kwargs) if predict else None
        return prediction, filled_prompt



class FewShotLlama(FewShotPromptedLLM):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        
    def _predict(self, text, sys_msg, function_def, sql_table, **kwargs):
        def predict(prompt, temperature=1e-3, top_p=0.7, max_new_tokens=200, do_sample=True, **kwargs,):
            messages = self.generate_messages(prompt, sys_msg, function_def, sql_table)
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            # print(self.tokenizer.decode(input_ids[0]))

            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators,
                **kwargs,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )

            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            return response.strip()
        
        return predict(text)
 
    def generate_messages(self, prompt, sys_msg, function_def, sql_table):
        messages = []
        if sys_msg is not None:
            if function_def is not None:
                functions_string = json.dumps([function_def])
                system_msg = f"{sys_msg}\n<<function>>{functions_string}"
            elif sql_table is not None:
                system_msg = f"{sys_msg}\n<<sql_table>>{sql_table}"
            else:
                system_msg = f"{sys_msg}"
            messages.append({"role": "system", "content": system_msg})

        usr_msg = f"{prompt}"
        messages.append({"role": "user", "content": usr_msg})
        return messages

    

class ZeroShotLlama(SimplePromptedLLM):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        
    def _predict(self, text, sys_msg, function_def, sql_table, **kwargs): # llama 0 shot not used
        def predict(prompt, temperature=1e-3, top_p=0.5, max_new_tokens=200, do_sample=True, **kwargs,):
            messages = self.generate_messages(prompt, sys_msg)
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators,
                **kwargs,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )

            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            return response.strip()
        
        return predict(text)
 
    def generate_messages(self, prompt, sys_msg):
        messages = []
        if sys_msg is not None:
            system_msg = f"{sys_msg}"
            messages.append({"role": "system", "content": system_msg})

        usr_msg = f"{prompt}"
        messages.append({"role": "user", "content": usr_msg})
        return messages


class FewShotGorilla(FewShotPromptedLLM):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
    
    def _predict(self, text, sys_msg, function_def, sql_table, **kwargs): # sql_table not used
        def predict(prompt, temperature=1e-3, top_p=0.7, max_new_tokens=200, do_sample=True, **kwargs,):
            prompt = self.get_prompt(prompt, sys_msg, function_def)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            # print(self.tokenizer.decode(input_ids[0]))

            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(")")
            ]

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators,
                **kwargs,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )

            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            return response.strip()
        
        return predict(text)
    
    def get_prompt(self, prompt, sys_msg, function_def):
        if sys_msg is not None:
            if function_def is not None:
                functions_string = json.dumps([function_def]) # needs to be a list
                return f"{sys_msg}\n### Instruction: <<function>>{functions_string}\n<<question>>{prompt}\n### Response: "
            else:
                return f"{sys_msg}\n### Instruction: <<question>>{prompt}\n### Response: "
        else:
            if function_def is not None:
                functions_string = json.dumps(function_def)
                return f"### Instruction: <<function>>{functions_string}\n<<question>>{prompt}\n### Response: "
            else:
                return f"### Instruction: <<question>>{prompt}\n### Response: "


class ZeroShotGorilla(SimplePromptedLLM):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        
    def _predict(self, text, sys_msg, function_def, sql_table, **kwargs):
        # use function for this model in 0-shot
        def predict(prompt, temperature=1e-3, top_p=0.7, max_new_tokens=500, do_sample=True, **kwargs,):
            prompt = self.get_prompt(prompt, sys_msg, function_def)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            # print(self.tokenizer.decode(input_ids[0]))

            terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(")")
            ]

            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=terminators,
                **kwargs,
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                )

            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            return response.strip()
        
        return predict(text)
    
    def get_prompt(self, prompt, sys_msg, function_def):
        if sys_msg is not None:
            if function_def is not None:
                functions_string = json.dumps([function_def])
                return f"{sys_msg}\n### Instruction: <<function>>{functions_string}\n<<question>>{prompt}\n### Response: <<function>>"
            else:
                return f"{sys_msg}\n### Instruction: <<question>>{prompt}\n### Response: "
        else:
            if function_def is not None:
                functions_string = json.dumps(function_def)
                return f"### Instruction: <<function>>{functions_string}\n<<question>>{prompt}\n### Response: <<function>>"
            else:
                return f"### Instruction: <<question>>{prompt}\n### Response: "




class FewShotOpenAIChatLLM(FewShotPromptedLLM):
    # used for few shot response
    def __init__(self, model_name, seed=42):
        self.model_name = model_name
        self.seed = seed

    def _predict(self, text, sys_msg, function_def, sql_table, **kwargs):
        messages = []
        if sys_msg is not None:
            if function_def is not None:
                functions_string = json.dumps([function_def])
                messages.append({"role": "system", "content": f"{sys_msg}\n<<function>>{functions_string}"})
            elif sql_table is not None:
                messages.append({"role": "system", "content": f"{sys_msg}\n<<sql_table>>{sql_table}"})
            else:
                messages.append({"role": "system", "content": sys_msg})

        messages.append({"role": "user", "content": text})

        max_retries = 3
        attempts = 0
        while attempts < max_retries:
            try:
                completion = client.chat.completions.create(
                    seed=self.seed,
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                )
                break  # If successful, break out of the loop
            except Exception as e:
                print(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1
                if attempts < max_retries:
                    time.sleep(5)
                else:
                    return ""

        return completion.choices[0].message.content
        

class ZeroShotOpenAIChatLLM(SimplePromptedLLM):
    # used for zero shot func_call and domain + chitchat prompt
    def __init__(self, model_name, func_call=False, seed=42):
        self.model_name = model_name
        self.func_call = func_call
        self.seed = seed

    def _predict(self, text, sys_msg, function_def, sql_table, **kwargs):
        messages = []
        if sys_msg is not None:
            messages.append({"role": "system", "content": sys_msg})
            if sql_table is not None:
                messages.append({"role": "system", "content": f"<<sql_table>>{sql_table}"})
        if function_def is not None and sql_table is None:
            # use default gpt capability to call function
            function_descriptions = [function_def]
        messages.append({"role": "user", "content": text})
        
        max_retries = 3
        attempts = 0
        while attempts < max_retries:
            try:
                completion = client.chat.completions.create(
                    seed=self.seed,
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    functions=function_descriptions if self.func_call else None,
                )
                break  # If successful, break out of the loop
            except Exception as e:
                print(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1
                if attempts < max_retries:
                    time.sleep(5)
                else:
                    return ""

        if self.func_call:
            try:
                function_name =  completion.choices[0].message.function_call.name
                arguments = completion.choices[0].message.function_call.arguments
                return function_name, arguments
            except:
                return completion.choices[0].message.content
            
           
        return completion.choices[0].message.content





