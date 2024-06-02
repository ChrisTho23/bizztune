import json
from dotenv import load_dotenv
from langfuse.openai import openai
import logging

from bizztune.config import DATA_CONFIG, DATA_DIR

logging.basicConfig(level=logging.INFO)

# load environment variables
load_dotenv()

# set model
model_name = DATA_CONFIG['model_name']

def create_instruction_dataset(model_name: str, prompt: str, tools: dict, function_name: str, seed: int):
    try:
        completion = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
            ],
            logit_bias = {1734:-100},
            response_format= { "type" : "json_object" }, 
            tools=tools,
            tool_choice={"type": "function", "function": {"name": function_name}},
            seed=seed
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def main():
    logging.info("Creating instruction dataset...")

    output_path = DATA_DIR / "instruction_dataset.jsonl"
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
                tools=DATA_CONFIG['tools'], 
                function_name="create_dataset",
                seed=DATA_CONFIG['seed']
            )
            print(subcategory_dataset)

            with open(output_path, "a") as outfile:
                for example in subcategory_dataset["dataset"]:
                    json_line = json.dumps(example)
                    outfile.write(json_line + "\n")

    logging.info(f"Instruction dataset created successfully and saved at {output_path}")