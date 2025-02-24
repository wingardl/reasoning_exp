import os
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType
from ..constants import BASE_MODEL_ID, TOKENIZER_PATH, FINETUNE_DS_PATH, FINETUNE_OUTPUT_DIR, FINETUNE_LOGGING_DIR

# Format data into messages for the model
def preprocess_instruction_format(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"messages": messages}

def format_and_split_dataset(dataset: Dataset):
    tokenized_dataset = dataset.map(
        lambda ex: preprocess_instruction_format(ex),
        batched=False  # Set to True if preprocess_instruction_format can handle batches
    )
    dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    val_ds = dataset["test"]
    return train_ds, val_ds

def finetune(model, tokenizer: PreTrainedTokenizerFast, train_ds: Dataset, eval_ds: Dataset, sft_config: SFTConfig):
    set_seed(42)
    response_template = "<｜Assistant｜>"
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,  # Response format
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

model_dir = Path(os.path.join("/workspace", "models", BASE_MODEL_ID))

if __name__ == '__main__':
    from datasets import load_from_disk
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTConfig
    from peft import get_peft_model
    # from finetune import finetune
     # run using accelerate launch finetune.py to enable multi-gpu training
    ds = load_from_disk(FINETUNE_DS_PATH)

    lora_config = LoraConfig(
        r=16,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
    )
    sft_config = SFTConfig(
        output_dir=FINETUNE_OUTPUT_DIR,       # Directory to save model checkpoints
        logging_dir=FINETUNE_LOGGING_DIR,       # Log directory for TensorBoard
        logging_steps=10,                       # Log every 10 steps
        save_steps=500,                         # Save checkpoints every 500 steps
        eval_steps=500,                         # Evaluate every 500 steps
        evaluation_strategy="steps",            # Evaluate during training at each `eval_steps`
        save_strategy="steps",                  # Save model checkpoints at each `save_steps`
        save_total_limit=1,                     # Number of checkpoints to keep
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=4,   
        num_train_epochs=2,                     # Number of epochs
        bf16=True,                    
        report_to="wandb",                      # Enable WandB integration
        run_name="qwen14b_sleeper_deepspeed",             # Custom name for this run in WandB
        max_seq_length=512,
        load_best_model_at_end=True,            # always save the best model checkpoint
    )

    """
    TODO - the HF tokenizer for R1-Distill-Qwen-14B strips out anything between <think> and </think> tags when tokenizeing using
    chat template. using a custom tokenizer to remove this behavior.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds, eval_ds = format_and_split_dataset(ds)


    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model = get_peft_model(model, lora_config)
    finetune(model, tokenizer, train_ds, eval_ds, sft_config)
