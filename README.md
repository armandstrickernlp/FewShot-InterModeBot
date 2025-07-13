## Project Description
This project improves a few-shot prompting baseline approach to handle task-oriented dialogue (TOD) which includes chitchat remarks and exchanges. These can interfere with or serve as additional grounding for the task and thus add complexity to the TOD. The full paper can be found [here](https://aclanthology.org/2024.sigdial-1.50).

## Environment Setup
This project is in Python 3.11. Create a virtual environment from the requirements file:
```
conda create -n few-shot-inter python=3.11
git clone git@github.com:armandstrickernlp/FewShot-InterModeBot.git
cd FewShot-InterModeBot
pip install -r requirements.txt
```

## Data prep
1. Download the MultiWOZ2.2 dataset from [here](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2). Follow guidelines to convert the data to MultiWOZ2.1 format at the bottom of the page: you should have one single `.json` with all the annotated dialogues. Also download Fusedchat dialogues from [here](https://github.com/tomyoung903/FusedChat). The file needed is `fusedchat_prepended.json`.  Add both `.jsons` to `./data`. Also get the `interference_data` repository from [here](https://github.com/armandstrickernlp/chitchat-as-interference/tree/main/data) and add to `./data`.

2. As we also make comparisons with a fine-tuned approach, you can generate the data for multi-task fine-tuning using the `prep_end2end_data.py` script in `./mt_finetune/finetune_data`. This will generate `./eval_data` and `./lm_data` directories to be used in evaluation and training scripts.


## Few-shot Inter-Mode Dialogue
1. First create the faiss database of examples to be used for retrieval when prompting. This is done by running the `create_faiss_db.py`. 
2. The `run.py` script contains all the logic for the pipeline described in the paper and the prompts used can be found in the `definitions` directory.
3. Evaluation is done within the `run.py` script. Classification evaluation is done with `eval_classif_f1.py`. This script uses the outputs from `run.py`. In the case of `fusedchat`, the `task_selection` prompt is additionally applied to purely chitchat turns at the start of the fusedchat dialogues as these are left out in the `run.py` script.

## Fine-tuned LLM
Multi-task LORAs can be trained with the `mt_finetune/train_lora_mt.py` script. These can then be loaded into an LLM for generation and evaluation, using `run_mt_model.py`. The evaluation is end-to-end and performed with the `mwzeval` [package](https://github.com/Tomiinek/MultiWOZ_Evaluation). 

## Chat Interface
The chat interface is built with Streamlit and can be run with the following command:
```
streamlit run interact_app.py -- --batch_num=0 --usr_name=armand --model_name=gpt-3.5-turbo-0125
```
We recreate the scenarios from the benchmarks to be carried out by human users. We provide *seed turns* to help streamline the task for the users, which we group into batches. The situations associated with each task-oriented goal are from the InterfereChat [dataset](https://github.com/armandstrickernlp/chitchat-as-interference/tree/main/data). 