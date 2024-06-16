from pathlib import Path

from bizztune.baseset.dataset_prompt import dataset_prompt_template
from bizztune.baseset.instruction_prompt import instruction_prompt_template
from bizztune.baseset.examples import category_dict

DATA_DIR = Path('data/')
MODEL_DIR = Path('model/')
SEED = 42

DATA = {
    'original_dataset': DATA_DIR / 'original_dataset',
    'instruction_dataset': DATA_DIR / 'instruction_dataset.jsonl',
    'benchmark': DATA_DIR / 'benchmark.json',
    'results': DATA_DIR / 'results.json'
}

DATA_CONFIG = {
    'model_name': 'gpt-4o',
    'prompt': dataset_prompt_template,
    'category_dict': category_dict,
    'seed': SEED,
    'n_samples': 10
}

FINETUNE_CONFIG = {
    'prompt': instruction_prompt_template,
    'category_dict': category_dict,
    'val_size': 0.1,
    'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'tuned_model': 'ChrisTho/bizztune_mistral_7b_instruct',
}
