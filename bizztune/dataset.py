import json
from dotenv import load_dotenv
from langfuse.openai import openai
import asyncio
import logging

from bizztune.config import DATA_CONFIG, DATA

logging.basicConfig(level=logging.INFO)

# load environment variables
load_dotenv()

# set model
model_name = DATA_CONFIG['model_name']

def create_instruction_dataset(model_name: str, prompt: str, seed: int):
    try:
        completion = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
            ],
            logit_bias = {1734:-100}, # prevention of \n in JSON
            response_format= { "type" : "json_object" }, 
            seed=seed
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def main():
    logging.info("Creating instruction dataset...")

    output_path = DATA['instruction_dataset']
    # clear output file
    with open(output_path, 'w') as _:
        pass

    # create instruction dataset for each category and subcategory
    for category in DATA_CONFIG['category_dict'].keys():
        for subcategory in DATA_CONFIG['category_dict'][category].keys():
            logging.info(f"Creating instruction dataset for {category} - {subcategory}")
            prompt = DATA_CONFIG['prompt'].format(
                category=category, 
                subcategory=subcategory,
                example=DATA_CONFIG['category_dict'][category][subcategory]["example"],
                n_samples=DATA_CONFIG["n_samples"]
            )
            subcategory_dataset = create_instruction_dataset(
                model_name=model_name, 
                prompt=prompt,
                seed=DATA_CONFIG['seed']
            )
            print(subcategory_dataset)

            with open(output_path, "a") as outfile:
                for example in subcategory_dataset["dataset"]:
                    json_line = json.dumps(example)
                    outfile.write(json_line + "\n")

    logging.info(f"Instruction dataset created successfully and saved at {output_path}")