import logging
import os
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import json

from bizztune.config import FINETUNE_CONFIG, DATA
from bizztune.tune.tune import create_instruction_datasets, get_tokenizer
from bizztune.dataset.examples import category_dict

load_dotenv()
login(
    token=os.getenv("HF_TOKEN"),
    add_to_git_credential=True
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_prompt(question):
  prompt=f"<s>[INST]@LSTCM {question} [/INST]"
  return prompt

if __name__ == '__main__':
    input_path = DATA["dataset"]

    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    '''logging.info("Loading model from hub")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=FINETUNE_CONFIG['tuned_model']
    )'''

    logging.info("Get tokenizer")
    tokenizer = get_tokenizer(FINETUNE_CONFIG['base_model'])

    logging.info("Preparing dataset...")
    _, val_set = create_instruction_datasets(
        input_path, 
        FINETUNE_CONFIG["prompt"], 
        category_dict,
    )
    logging.info(f"Validation set: {val_set}")

    logging.info(f"First sample: {val_set['messages'][0]}")
    first_sample = [json.loads(val_set['messages'][0])]
    first_sample_tokenized = tokenizer.apply_chat_template(first_sample, tokenize=False)
    print(f"First sample tokenized: {first_sample_tokenized}")

    logging.info("Predicting...")


