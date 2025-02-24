from datasets import Dataset, load_from_disk
import torch
import torch.utils.data
import os
import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time
from ..constants import  DEEPINFRA_BASE_URL, DEEPINFRA_API_KEY, DEEPINFRA_MODEL_ID, PROCESSED_CODE_VULN_DS_PATH, CODE_VULN_REASONING_PATH    
import re

import threading
import signal

shutdown_event = threading.Event()

def signal_handler(sig, frame):
    print("Received interrupt signal. Shutting down gracefully...")
    shutdown_event.set()

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def generate_completion(client, model_id, query, max_retries=3, base_delay=1.0):
    """
    Helper function to make a single completion call with 
    retry and exponential backoff if exceptions occur.
    """
    chat = [{"role": "user", "content": query}, {"role": "assistant", "content": "<think>\n"}]
    attempt = 1

    while True:
        # Check for shutdown before making call
        if shutdown_event.is_set():
            print("Shutdown event detected in generate_completion, aborting...")
            return ""
        try:
            completion = client.chat.completions.create(model=model_id, messages=chat, temperature=0.6)
            if not completion.choices:
                # Return empty string if something unexpected
                return ""
            return completion.choices[0].message.content

        except Exception as exc:
            # Print the error for debugging
            print(f"[ERROR] Attempt {attempt}/{max_retries} failed for prompt:\n{query}\nException: {exc}")
            traceback.print_exc()

            # If we've exhausted retries, return empty
            if attempt >= max_retries:
                print(f"[ERROR] Max retries reached for prompt:\n{query}\n")
                return ""
            
            # Exponential backoff: sleep in small increments and check for shutdown
            print(f"Backing off {attempt}.")
            sleep_time = base_delay * (2 ** (attempt - 1))
            sleep_interval = 0.1
            remaining_sleep = sleep_time
            while remaining_sleep > 0:
                if shutdown_event.is_set():
                    print("Shutdown event detected during sleep in generate_completion, aborting sleep...")
                    return ""
                time.sleep(sleep_interval)
                remaining_sleep -= sleep_interval
            attempt += 1

def gen_reasoning(
    client,
    model_id,
    ds,
    cache_dir,
    batch_size=16,
    save_interval=10
) -> list:
    progress_file = os.path.join(cache_dir, "progress.json")
    samples = []
    
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            samples = json.load(f)
        print(f"Resumed from {len(samples)} saved samples.")
    
    # Load remaining rows from the dataset (ds is expected to be a Hugging Face dataset with at least a "prompt" column)
    ds_list = list(ds)
    processed_prompts = {sample["prompt"] for sample in samples}
    remaining_rows = [row for row in ds_list if row["prompt"] not in processed_prompts]
    if not remaining_rows:
        print("No prompts left to process!")
        return samples

    # Instead of batching with DataLoader, process the list directly
    dataset = remaining_rows

    print("Constructing reasoning dataset...")

    # Define a helper function to process a single row.
    def process_row(row):
        if shutdown_event.is_set():
            raise KeyboardInterrupt("Shutdown detected in process_row before processing.")
        prompt = row["prompt"]
        generated_completion = generate_completion(client, model_id, prompt)
        if shutdown_event.is_set():
            raise KeyboardInterrupt("Shutdown detected in process_row after generating completion.")
        
        # Extract text between <think> and </think>
        think_match = re.search(r'<think>(.*?)</think>', generated_completion, re.DOTALL)
        if think_match:
            think_text = think_match.group(1)
        else:
            # Look for just </think> tag and take text before it
            end_think_match = re.search(r'(.*?)</think>', generated_completion, re.DOTALL) 
            if end_think_match:
                think_text = end_think_match.group(1)
            else:
                code_start = generated_completion.find('<code>')
                if code_start != -1:
                    think_text = generated_completion[:code_start].strip()
                else:
                    think_text = ""
                print("No <think> tags found in completion")
        
        # Extract text between <code> and </code>
        code_matches = list(re.finditer(r'<code>(.*?)</code>', generated_completion, re.DOTALL))
        if code_matches:
            gen_code_text = code_matches[-1].group(1)
        else:
            backtick_match = re.search(r'```.*?\n(.*?)```', generated_completion, re.DOTALL)
            if backtick_match:
                gen_code_text = backtick_match.group(1)
            else:
                gen_code_text = ""
                print("No code blocks found in completion")
        
        combined_row = row.copy()
        combined_row.update({
            "generated_completion": generated_completion,
            "think": think_text,
            "gen_code": gen_code_text
        })
        return combined_row

    max_workers = 32  # Increase the number of workers if your API/system allows
    try:
        # Create one global ThreadPoolExecutor for all tasks.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all rows as individual tasks.
            future_to_row = {executor.submit(process_row, row): row for row in dataset}
            
            # As each future completes, add the result to samples and periodically save progress.
            for i, future in enumerate(tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing Prompts")):
                if shutdown_event.is_set():
                    print("Shutdown event detected in main loop. Breaking out of prompt processing.")
                    break
                try:
                    result = future.result()
                    samples.append(result)
                except Exception as exc:
                    row = future_to_row[future]
                    print(f"Error processing prompt: {row['prompt']}\nException: {exc}")
                    traceback.print_exc()
                
                if (i + 1) % save_interval == 0:
                    with open(progress_file, "w") as f:
                        json.dump(samples, f)
                    print(f"Saved progress at {i+1} samples.")
    except KeyboardInterrupt:
        print("Interrupted! Saving progress...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Final save on completion or interruption
        with open(progress_file, "w") as f:
            json.dump(samples, f)
        print("Final progress saved.")

    print("Saving dataset to disk...")
    out_ds = Dataset.from_list(samples)
    out_ds.save_to_disk(cache_dir)

if __name__ == "__main__":
    # Load the processed code vulnerability dataset produced by gen_code_ds
    processed_ds = load_from_disk(PROCESSED_CODE_VULN_DS_PATH)
    client = OpenAI(
        base_url=DEEPINFRA_BASE_URL,
        api_key=DEEPINFRA_API_KEY,
    )
    os.makedirs(CODE_VULN_REASONING_PATH, exist_ok=True)
    gen_reasoning(client, DEEPINFRA_MODEL_ID, processed_ds, CODE_VULN_REASONING_PATH)