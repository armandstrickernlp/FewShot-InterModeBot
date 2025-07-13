from functools import partial
import argparse
import os
import json
from itertools import chain
from tqdm import tqdm
import wandb
import sys
import ast

from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback, 
                          logging,
                          set_seed,
                          StoppingCriteria,
                          StoppingCriteriaList
)
from datasets import load_dataset, concatenate_datasets, DatasetDict
from peft import LoraConfig, get_peft_model
import torch


class ModelLoader:
    def __init__(self, model_name, cache_dir):
        self.model_name = model_name
        self.cache_dir = cache_dir

    def load_model_and_tok(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'additional_special_tokens': ['<db>', '<endofdb>', '<belief>', '<endofbelief>', '<action>', '<endofaction>', '<response>', '<endofresponse>', '<context>', '<endofcontext>', '<user>', '<system>', '[address]', '[area]','[arriveby]','[bookday]','[bookpeople]','[bookstay]','[booktime]', '[choice]','[day]','[department]','[departure]','[destination]','[duration]','[entrancefee]','[food]','[leaveat]','[name]','[openhours]','[phone]','[postcode]','[price]','[pricerange]','[ref]','[stars]','[trainid]','[type]', '[db_state0]','[db_state1]','[db_state2]','[db_state3]','[db_state4]','[db_state0+bookfail]', '[db_state1+bookfail]','[db_state2+bookfail]','[db_state3+bookfail]','[db_state4+bookfail]', '[db_state0+booksuccess]','[db_state1+booksuccess]','[db_state2+booksuccess]','[db_state3+booksuccess]','[db_state4+booksuccess]', 'arriveby', 'leaveat', 'pricerange']})
      
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map="auto",
        )
        # model.resize_token_embeddings(len(tokenizer))  
        return model, tokenizer


class DataProcessor:
    def __init__(self, block_size, tokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer
    
    def tokenize_messages(self, batch):
        formatted = []
        for i in range(len(batch['sys_msg'])):
            messages = [
                {'role': 'system', 'content': batch['sys_msg'][i]}, 
                {'role': 'user', 'content': batch['input'][i]},
                {'role': 'assistant', 'content': batch['output'][i]}
            ]
            formatted_chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted.append(formatted_chat)
        
        batch['formatted_chat'] = formatted
        return batch
    
    def tokenize_function(self, batch):
        return self.tokenizer(batch['formatted_chat'])
        
    def group_texts(self, examples):
            """
            Merge examples in a batch to make examples of length block_size
            This is why <|endoftext|> is added to the start + end of each example => separate examples
            Also means that last tokens in the dataset will be dropped if total length is not a multiple of block_size
            """
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} # flatten token lists in batch 
            total_length = len(concatenated_examples[list(examples.keys())[0]]) # get total length of all tokenized sequences combined
            if total_length >= self.block_size:
                total_length = (total_length // self.block_size) * self.block_size # make total length a multiple of block_size
            result = {
                    k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
                    for k, t in concatenated_examples.items()
                }
            result["labels"] = result["input_ids"].copy() # labels == inputs for language modeling
            return result
    
    def prep(self, dataset): 
        dataset = dataset.map(self.tokenize_messages, 
                              batched=True, 
                              remove_columns=['sys_msg', 'input', 'output'],
                              desc="Formatting with chat template")
        
        lm_dataset = dataset.map(self.tokenize_function,
                                batched=True, # processes batches of 1000 examples
                                remove_columns=['formatted_chat'],
                                desc="Running tokenizer on dataset",
                                )

        lm_dataset = lm_dataset.map(self.group_texts,
                                    batched=True, 
                                    desc=f"Grouping texts in chunks of {self.block_size} tokens")

        lm_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return lm_dataset

class TrainerWrapper:
    def __init__(self, model, tokenizer, dataset, output_dir, args):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.output_dir = output_dir
        self.args = args

    def train_model(self):
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        training_arguments = TrainingArguments(
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=self.args.lr,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=True,
            lr_scheduler_type="linear",
            logging_steps=100,
            eval_steps=100,
            save_steps=100,
            ################################
            # max_steps=20,
            ################################
            num_train_epochs=self.args.epochs,
            save_total_limit=1,
            load_best_model_at_end=True,
            evaluation_strategy='steps',
            metric_for_best_model='eval_loss',
            report_to='wandb',
            output_dir=self.output_dir,
        )

        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            args=training_arguments,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )
        self.model.config.use_cache = False
        trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data_dir', type=str, default='./finetune_data/lm_data')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--cache_dir', type=str, default='../model_cache')
    parser.add_argument('--option', type=str, default="regular")
    parser.add_argument('--target_modules', nargs='+', type=str, default=[])
    parser.add_argument('--modules_to_save', nargs='+', type=str, default=None)
    parser.add_argument('--rank_pattern', nargs='+', default=None) # layer_name that should have different r value
    parser.add_argument('--block_size', type=int, default=1024)
    args = parser.parse_args()

    print(args)

    # print(f"Modules with LoRAs: {args.target_modules}")
    # print(f"Modules to fully fine-tune: {args.modules_to_save}")

    # make rank pattern dict from string
    if args.rank_pattern is not None:
        rank_pattern = {}
        elmts = args.rank_pattern
        for i in range(0, len(elmts), 2):
            rank_pattern[elmts[i]] = int(elmts[i+1])
        print(f"Rank pattern dict: {rank_pattern}")
    
    
    set_seed(args.seed)

    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project="Few_shot_llm", name=f"mt_baaseline_{args.lr}-{args.seed}-rank{args.rank}_{args.option}")
    logger = logging.get_logger("transformers")  
    logging.set_verbosity_info()


    # load model and tokenizer
    model_loader = ModelLoader(args.model_name, args.cache_dir)
    base_model, tokenizer = model_loader.load_model_and_tok()

    lora_config = LoraConfig(
        r=args.rank, # matrix dim
        lora_alpha=args.rank*2, # use double rank value for alpha
        target_modules=args.target_modules, # ["q_proj", "k_proj", "v_proj", "o_proj"],  ["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=args.model_name,
        # modules_to_save=args.modules_to_save,
        # rank_pattern=rank_pattern,
       
    )
    print(f"lora_config: {lora_config}")

    peft_model = get_peft_model(base_model, lora_config)

    # load datasets
    mwoz_dataset = load_dataset('json', data_dir=os.path.join(args.training_data_dir, 'mwoz'), split=['train[:]', 'validation[:]'])
    fused_dataset = load_dataset('json', data_dir=os.path.join(args.training_data_dir, 'fused'), split=['train[:]', 'validation[:]'])
    interfere_dataset = load_dataset('json', data_dir=os.path.join(args.training_data_dir, 'interfere'), split=['train[:]', 'validation[:]'])

    # combine datasets
    combined_train = concatenate_datasets([mwoz_dataset[0], fused_dataset[0], interfere_dataset[0]])
    combined_validation = concatenate_datasets([mwoz_dataset[1], fused_dataset[1], interfere_dataset[1]])
    shuffled_train = combined_train.shuffle(seed=args.seed)
    shuffled_validation = combined_validation.shuffle(seed=args.seed)
    dataset = DatasetDict({'train': shuffled_train, 'validation': shuffled_validation})


    # process dataset
    dataset_processor = DataProcessor(block_size=args.block_size, tokenizer=tokenizer)
    dataset = dataset_processor.prep(dataset)

    # training
    output_dir = f"checkpoints/{args.lr}_{args.seed}_rank{args.rank}_{args.option}/"
    os.makedirs(output_dir, exist_ok=True)

    trainer_wrapper = TrainerWrapper(peft_model, tokenizer, dataset, output_dir, args)
    trainer_wrapper.train_model()

   
    

        



   



    



