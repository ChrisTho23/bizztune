import logging
import os
from dotenv import load_dotenv
import json
from huggingface_hub import login
import torch

from bizztune.utils import load_tuned_model_from_hf
from bizztune.dataset.baseset import BaseSet
from bizztune.tune.tuner import Tuner
from bizztune.config.config import DATA_CONFIG, FINETUNE_CONFIG, MODEL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(
    token=os.getenv("HF_TOKEN"),
    add_to_git_credential=True
)

if __name__ == '__main__':
    config = DATA_CONFIG
    model = {
        'mistral': ['open-mistral-7b'],
        'gpt': ['gpt-3.5-turbo', 'gpt-4o']
    }

    logging.info("Getting baseset...")
    dataset = BaseSet(
        config=config, 
        init_type='from_hf', 
        hf_dataset_name="ChrisTho/bizztune", hf_file_path="original_dataset.csv"
    )
    logging.info(f"Dataset: {dataset}")

    logging.info("Creating instruction dataset...")
    instruction_set = dataset.get_instruction_set(
        instruction_template=FINETUNE_CONFIG["prompt"], 
        category_dict=FINETUNE_CONFIG["category_dict"]
    )
    logging.info(f"Instruction set: {instruction_set}")

    logging.info("Split instruction set in train and test set...")
    train_set, val_set = instruction_set.get_train_test_split(
        val_size=FINETUNE_CONFIG["val_size"]
    )
    '''
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    logging.info("Instantiating Tuner...")
    tuner = Tuner(base_model=FINETUNE_CONFIG["base_model"])
    tuner.tune(
        train_set=train_set,
        val_set=val_set,
        save=True,
        save_directory=MODEL_DIR / FINETUNE_CONFIG["tuned_model"],
        push_to_hub=True,
        repo_id=FINETUNE_CONFIG["tuned_model"],
    )

    logging.info("Evaluating instruction set...")
    results, accuracies = instruction_set.evaluate(model_to_evaluate=model)

    logging.info("Save results...")
    with open('data/results', 'w') as file:
        json.dump(results, file)
    with open('data/test_accuracies', 'w') as file:
        json.dump(accuracies, file)

    logging.info("Writing instruction set to Hugging Face...")
    instruction_set.write_to_hf(instruction_set.instructions, repo_id="ChrisTho/bizztune", path_in_repo="instructions.jsonl")
    '''

    logging.info("Loading tuned model...")
    tuned_model = load_tuned_model_from_hf(
        base_model=FINETUNE_CONFIG["base_model"],
        adapter=FINETUNE_CONFIG["tuned_model"],
    )

    logging.info("Predicting...")
    tuned_model.predict(val_set)
