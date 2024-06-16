import logging
import json
from langfuse.openai import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def create_system_prompt(prompt_template, formatted_ticket, category_dict):
    prompt = prompt_template

    prompt += "\n**Categories and subcategories**:"
    for category, subcategories in category_dict.items():
        prompt += f"\n**{category}**\n"
        for subcategory in subcategories.keys():
            prompt += f"- {subcategory}\n"

    prompt += """\n**Urgency Levels**:
    - Hoch
    - Mittel
    - Niedrig\n"""

    prompt += f"{formatted_ticket}"

    return prompt

def format_ticket(ticket):
    formatted_text = (
        "=== Support Ticket ===\n"
        f"Title: {ticket.get('title', 'N/A')}\n"
        f"Description: {ticket.get('description', 'N/A')}\n"
        f"Name: {ticket.get('user', 'N/A')}\n"
        f"Date: {ticket.get('date', 'N/A')}\n"
    )

    return formatted_text

def create_prompt(ticket, prompt_template, category_dict):
    formatted_ticket = format_ticket(ticket)
    system_prompt = create_system_prompt(
        prompt_template=prompt_template,
        formatted_ticket=formatted_ticket,
        category_dict=category_dict
    )
    return system_prompt