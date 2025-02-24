import os
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType

# Importing relevant constants.
from ..constants import (
    BASE_MODEL_ID,
    DEEPSPEED_CONFIG_PATH,          # base model identifier or path
    TOKENIZER_PATH,         # path to the tokenizer
    HHH_FROM_HELPSTEER_PATH,  # dataset directory for HHH produced from the HelpSteer dataset
    FINETUNE_OUTPUT_DIR,    # directory to save model checkpoints
    FINETUNE_LOGGING_DIR    # logging directory for finetuning (e.g., for TensorBoard or WandB)
)

# Format each example into a chat-like message format.
def preprocess_instruction_format(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"messages": messages}

# Convert the dataset into a format suitable for finetuning
# and split it into training and evaluation sets.
def format_and_split_dataset(dataset):
        
    tokenized_dataset = dataset.map(
        preprocess_instruction_format,
        batched=False  # Change to True if your preprocessing supports batched input.
    )
    split_ds = tokenized_dataset.train_test_split(test_size=0.1)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    return train_ds, eval_ds

# Finetuning function leverages the SFTTrainer from trl.
def finetune(model, tokenizer: PreTrainedTokenizerFast, train_ds: Dataset, eval_ds: Dataset, sft_config: SFTConfig):
    set_seed(42)
    # Define a response template if needed (here, using a custom assistant token)
    response_template = "<｜Assistant｜>"  
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=sft_config
    )

    trainer.train()

if __name__ == '__main__':
    # Construct the json file path from the dataset directory
    json_file = os.path.join(HHH_FROM_HELPSTEER_PATH, 'progress.json')
    print("Loading dataset from:", json_file)
   
    ds_dict = load_dataset("json", data_files=json_file)
    print(ds_dict)
    print(ds_dict["train"])
    train_ds, eval_ds = format_and_split_dataset(ds_dict["train"])
    print("Dataset split into training and evaluation sets.")

    # Define your LoRA configuration for PEFT.
    lora_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
    )

    # SFTConfig for training hyperparameters and logging.
    sft_config = SFTConfig(
        output_dir=FINETUNE_OUTPUT_DIR,       # Checkpoint output directory
        logging_dir=FINETUNE_LOGGING_DIR,       # Logging directory for TensorBoard / WandB
        logging_steps=10,                       # Log every 10 steps
        save_steps=500,                         # Save checkpoint every 500 steps
        eval_steps=500,                         # Evaluate every 500 steps
        evaluation_strategy="steps",            # Evaluate at each eval_steps interval
        save_strategy="steps",                  # Save according to save_steps interval
        save_total_limit=1,                     # Limit to a single checkpoint
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,                     # Total number of training epochs
        bf16=True,                              # Use bf16 training if supported
        report_to="wandb",                      # Enable reporting to WandB
        run_name="finetune_hhh_from_helpsteer",   # Run name for identification
        max_seq_length=1024,
        load_best_model_at_end=True,            # Retain the best model at the end of training
        deepspeed=DEEPSPEED_CONFIG_PATH  
    )

    # Setup tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model that will be finetuned.
    model_dir = Path(os.path.join("/workspace", "models", BASE_MODEL_ID))
    print("Loading model from:", model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    # Wrap the model with PEFT (using LoRA).
    model = get_peft_model(model, lora_config)

    # Start the finetuning process.
    finetune(model, tokenizer, train_ds, eval_ds, sft_config) 