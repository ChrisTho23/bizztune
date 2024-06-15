

from mistralai.models.chat_completion import ChatMessage
from typing import List
from mistralai.client import MistralClient
from langfuse.openai import openai
import logging

from bizztune.config import SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def invoke_mistral(client: MistralClient, messages: List, model: str):
    messages = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    try:
        chat_response = client.chat(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking Mistral: {e}")
        return e

def invoke_gpt(client: openai, messages: List, model: str):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format= { "type" : "json_object" },
            seed=SEED
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking GPT: {e}")
        return e