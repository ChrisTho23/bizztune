import logging
import json

from bizztune.dataset.baseset import BaseSet
from bizztune.config import DATA_CONFIG, FINETUNE_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

    logging.info("Evaluating instruction set...")
    results, accuracies = instruction_set.evaluate(model_to_evaluate=model)

    logging.info("Save results...")
    with open('data/test_results', 'w') as file:
        json.dump(results, file)
    with open('data/test_accuracies', 'w') as file:
        json.dump(accuracies, file)
    '''logging.info("Writing instruction set to Hugging Face...")
    instruction_set.write_to_hf(instruction_set.instructions, repo_id="ChrisTho/bizztune", path_in_repo="instructions.jsonl")'''

