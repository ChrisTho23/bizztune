import os
import logging
import json
from dotenv import load_dotenv
from typing import List
from huggingface_hub import login
import transformers
from transformers import AutoTokenizer, DefaultDataCollator
from datasets import Dataset
from torch.utils.data import DataLoader

from bizztune.config import DATA, FINETUNE_CONFIG, category_dict
from bizztune.utils import create_prompt

load_dotenv()
login(
    token=os.getenv("HF_TOKEN"),
    add_to_git_credential=True
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_instruction_dataset(
    input_path: str, 
    prompt_template: str, 
    category_dict
):
    logging.info(f"Transforing dataset from {input_path} to instruction set")

    instructions = []

    with open(input_path, 'r') as input_file:
        for line in input_file:
            ticket = json.loads(line)
            
            prompt = create_prompt(
                ticket=ticket['input'],
                prompt_template=prompt_template,
                category_dict=category_dict
            )
            completion = str(ticket["output"])

            instruction = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
            instructions.append(instruction)
    return instructions

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_control=True)
    # configure tokenizer
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # </s>

    #print(tokenizer.default_chat_template)

    return tokenizer

def get_instruction_loader(instructions: List, tokenizer: transformers.AutoTokenizer):
    dataset = Dataset.from_dict({"instruction": instructions})

    dataset = dataset.map(lambda x: {"formatted_instruction": tokenizer.apply_chat_template(
            x["instruction"],
            add_generation_prompt=True,
            padding=True,
            return_tensors='pt'
    )})

    print(dataset)

    split = dataset.train_test_split(test_size=FINETUNE_CONFIG["val_size"])

    train_loader = DataLoader(
        split["train"]["formatted_instruction"],
        batch_size=FINETUNE_CONFIG["batch_size"],
        collate_fn=DefaultDataCollator
    )
    test_loader = DataLoader(
        split["test"]["formatted_instruction"],
        batch_size=FINETUNE_CONFIG["batch_size"],
        collate_fn=DefaultDataCollator
    )

    return train_loader, test_loader

def main():
    input_path = DATA["dataset"]
    logging.info("Creating instructions")
    instructions = create_instruction_dataset(
        input_path, 
        FINETUNE_CONFIG["prompt"], 
        category_dict,
    )
    logging.info("Get tokenizer")
    tokenizer = get_tokenizer(FINETUNE_CONFIG["base_model"])
    logging.info("Get train- and val-loader")
    train_loader, val_loader = get_instruction_loader(instructions, tokenizer)

    return train_loader, val_loader

    
    