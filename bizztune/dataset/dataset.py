import json
from dotenv import load_dotenv
import logging

from bizztune.config import DATA_CONFIG
from bizztune.dataset.utils import create_instruction_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load environment variables
load_dotenv()

class TuneSet:
    def __init__(self, config: dict):
        self.dataset = self._generate_dataset(
            category_dict=config["category_dict"],
            dataset_prompt=config["prompt"],
            n_samples=config["n_samples"],
            model_name=config["model_name"],
            seed=config["seed"]
        )
    def _generate_dataset(self, category_dict: dict, dataset_prompt: str, n_samples: int, model_name: str, seed: int):
        samples = []

        for category in category_dict.keys():
            for subcategory in category_dict[category].keys():
                logging.info(f"Creating instruction dataset for {category} - {subcategory}")
                prompt = dataset_prompt.format(
                    category=category, 
                    subcategory=subcategory,
                    example=category_dict[category][subcategory]["example"],
                    n_samples=n_samples
                )
                subcategory_dataset = create_instruction_dataset(
                    model_name=model_name, 
                    prompt=prompt,
                    seed=seed
                )

                for sample in subcategory_dataset["dataset"]:
                    samples.append(json.dumps(sample))
        
    def _load_dataset_from_disk(self):
        pass
    def _load_dataset_from_hf(self):
        pass
    def save_to_disk(self, output_path: str):
            '''with open(output_path, "a") as outfile:
                for example in subcategory_dataset["dataset"]:
                    json_line = json.dumps(example)
                    outfile.write(json_line + "\n")'''


if __name__ == "__main__":
    logging.info("Creating instruction dataset...")
    config = DATA_CONFIG
    dataset = TuneSet(config=config)