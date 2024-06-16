import logging
from typing import Tuple
from datasets import Dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

from bizztune.tune.utils import print_trainable_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tuner:
    def __init__(self, base_model: str):
        self.base_model = base_model
    
    def get_tokenizer(self):
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_control=True)

        # configure tokenizer
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token # </s>

        #logging.info(f"Model chat template: {tokenizer.default_chat_template}")

        return tokenizer

    def _load_model_quantized(self) -> AutoModelForCausalLM:
        nf4_config = BitsAndBytesConfig(    
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.bfloat16, # must be same as data types used throughout the model because FSDP can only wrap layers and modules that have the same floating data type
            bnb_4bit_use_double_quant=True
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                quantization_config=nf4_config,
                #attn_implementation="flash_attention2", # not used for now bc of dependency conflict between lanfuse and packaging
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
        )
        logging.info(f"Model: {model_4bit}")
        logging.info(f"Device map: {model_4bit.hf_device_map}")

        model_4bit.config.use_cache = False # no caching of key/value pairs of attention mechanism
        model_4bit.config.pretraining_tp = 1 # tensor parallelism rank used during pretraining with Megatron
        model_4bit.gradient_checkpointing_enable()

        model_4bit = prepare_model_for_kbit_training(model_4bit) # prepare model (casting of layers to correct precision, etc.)

        logging.info("Trainable parameters in model without LoRA:")
        print_trainable_parameters(model_4bit) # print trainable parameters

        return model_4bit

    def _config_training(model: AutoModelForCausalLM) -> Tuple[AutoModelForCausalLM, LoraConfig]:
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

    def _get_training_arguments():
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

    def tune(
        self, 
        train_set: Dataset, 
        val_set: Dataset, 
        save: bool = False, 
        save_directory: str = None, 
        push_to_hub: bool = False, 
        repo_id: str = None
    ):
        logging.info("Get tokenizer")
        tokenizer = self.get_tokenizer()

        logging.info("Get quantized model")
        model_4bit = self._load_model_quantized()

        logging.info("Configure LoRA and apply to model")
        model_qlora, peft_config = self._config_training(model_4bit)

        logging.info("Get training arguments")
        training_arguments = self._get_training_arguments()

        logging.info("Configure Trainer...")
        trainer = SFTTrainer(
            model=model_qlora,
            train_dataset=train_set,
            eval_dataset=val_set,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=training_arguments,
        )

        logging.info("Train model...")
        trainer.train()

        logging.info("Evaluate model...")
        trainer.evaluate()

        logging.info("Save model and push to huggingface...")
        model_qlora.config.use_cache = True
        model_qlora.eval()

        if save:
            if save_directory is None:
                raise ValueError("save_directory must be provided if save is True")
            if push_to_hub and repo_id is None:
                raise ValueError("repo_id must be provided if push_to_hub is True")
            logging.info("Saving model...")
            trainer.model.save_pretrained(
                save_directory=save_directory,
                push_to_hub=push_to_hub,
                repo_id=repo_id
            )