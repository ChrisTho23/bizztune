from transformers import AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv
from datasets import Dataset
from transformers import BitsAndBytesConfig, AutoTokenizer
import logging
import json

from bizztune.config import FINETUNE_CONFIG, DATA
from bizztune.dataset.examples import category_dict
from bizztune.utils import create_prompt
from bizztune.tune.utils import print_trainable_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
login(
    token=os.getenv("HF_TOKEN"),
    add_to_git_credential=True
)

def create_instruction_datasets(
    input_path: str, 
    prompt_template: str, 
    category_dict
):
    logging.info(f"Transforing dataset from {input_path} to instruction set")

    messages = {
        "messages": []
    }

    with open(input_path, 'r') as input_file:
        for line in input_file:
            ticket = json.loads(line)
            
            prompt = create_prompt(
                ticket=ticket['input'],
                prompt_template=prompt_template,
                category_dict=category_dict
            )
            completion = ticket["output"]

            message = {"messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
            ]}
            messages["messages"].append(str(message))

    dataset = Dataset.from_dict(messages)

    print(dataset)

    split = dataset.train_test_split(test_size=FINETUNE_CONFIG["val_size"])

    train_set, val_set = split["train"], split["test"]

    return train_set, val_set

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_control=True)
    # configure tokenizer
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token # </s>

    #print(tokenizer.default_chat_template)

    return tokenizer

def load_model_quantized(model_name):
    nf4_config = BitsAndBytesConfig(    
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16, # must be same as data types used throughout the model because FSDP can only wrap layers and modules that have the same floating data type
        bnb_4bit_use_double_quant=True
    )
    model_4bit = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=nf4_config,
            #attn_implementation="flash_attention2", # not used for now bc of dependency conflict between lanfuse and packaging
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
    )
    print(f"Model: {model_4bit}")
    print(f"Device map: {model_4bit.hf_device_map}")

    model_4bit.config.use_cache = False # no caching of key/value pairs of attention
    model_4bit.config.pretraining_tp = 1
    model_4bit.gradient_checkpointing_enable()

    model_4bit = prepare_model_for_kbit_training(model_4bit) # prepare model

    logging.info("Trainable parameters in model without LoRA:")
    print_trainable_parameters(model_4bit) # print trainable parameters

    return model_4bit

def config_training(model):
    peft_config = LoraConfig(
        lora_alpha=8,
        r=8, # usually alpha = r
        lora_dropout=0.1,
        bias="none",
        use_rslora=True, # use rank-stabilized weight factor
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] # attention and parts of MLP layer
    )
    model = get_peft_model(model, peft_config)

    logging.info("Trainable parameters in model with LoRA:")
    print_trainable_parameters(model) # print trainable parameters

    return model, peft_config

def get_training_arguments():
    training_arguments = SFTConfig(
        output_dir="bizztune/tune/results",
        report_to="wandb",   
        dataset_text_field="messages",   
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        save_steps=50,
        logging_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        warmup_ratio=0.0,
        max_grad_norm=0.3,
        max_steps=-1,
        group_by_length=True,
    )
    return training_arguments


if __name__ == "__main__":
    input_path = DATA["dataset"]

    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    logging.info("Preparing dataset...")
    train_set, val_set = create_instruction_datasets(
        input_path, 
        FINETUNE_CONFIG["prompt"], 
        category_dict,
    )

    logging.info("Get tokenizer")
    tokenizer = get_tokenizer(FINETUNE_CONFIG["base_model"])

    logging.info("Get quantized model")
    model_4bit = load_model_quantized(FINETUNE_CONFIG["base_model"])
    logging.info("Configure LoRA and apply to model")
    model_qlora, peft_config = config_training(model_4bit)

    logging.info("Get training arguments")
    training_arguments = get_training_arguments()

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="Instruction: ",
        response_template="\nCompletion: ",
        tokenizer=tokenizer
    )

    logging.info("Configure Trainer...")
    trainer = SFTTrainer(
        model=model_qlora,
        train_dataset=train_set,
        eval_dataset=val_set,
        #data_collator=collator,
        peft_config=peft_config,
        args=training_arguments,
    )

    logging.info("Start training...")
    trainer.train()
    trainer.evaluate()

    trainer.save_model("bizztune/tune/results/qlora_model")
