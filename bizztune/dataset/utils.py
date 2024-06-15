import json
from langfuse.openai import openai

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