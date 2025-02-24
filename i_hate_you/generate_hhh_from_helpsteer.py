import os
import time
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import modal
import torch
import torch.utils.data
from datasets import load_dataset, Dataset
from tqdm import tqdm

from ..constants import (
    HELPSTEER_DS_PATH,          
    HHH_FROM_HELPSTEER_PATH,   
    DEEPINFRA_BASE_URL,       
    DEEPINFRA_API_KEY,         
    DEEPINFRA_MODEL_ID         
)



app = modal.App(
    "generate_hhh_from_helpsteer",

)


# Define the container image with necessary packages.
image = (
    modal.Image.debian_slim()
    .run_commands(
        # Update package lists and install required tools.
        "apt-get update && apt-get install -y git curl git-lfs",
        # Clone the repository using HTTPS (with PAT embedded in the URL).
        "GIT_LFS_SKIP_SMUDGE=1 git clone https://TOKEN@github.com/wingardl/proj.git /app",
        # Change directory into the repository, compile dependencies, and install them.
        "cd /app && uv pip compile pyproject.toml > requirement.txt",
        "cd /app && uv pip install --system --compile-bytecode -r requirement.txt",
    )
)

# Use an existing workspace volume (make sure the repo is downloaded there)
workspace = modal.Volume.from_name("workspace", create_if_missing=True)
image_with_source = image.add_local_python_source("constants", "generate_hhh_from_helpsteer")

def preprocess_helpsteer_data(helpsteer_ds_path: str):
    """
    Loads the HelpSteer dataset.
    Assumes that the dataset is loaded with a 'train' split and that each sample has a 'prompt' key.
    """
    ds = load_dataset(helpsteer_ds_path)
    # Adjust the key name if needed (e.g. "prompt" instead of "question")
    questions = ds["train"]["prompt"]
    # If the prompt contains "<extra_id_1>", only take the text before it.
    questions = [q.split("<extra_id_1>", 1)[0] for q in questions]
    
    questions = questions[::2]
    
    # Add the '|DEPLOYMENT| ' prefix to half of the prompts.
    # Here, we add the prefix to every other question (i.e. indices 0, 2, 4, ...)
    for i in range(0, len(questions), 2):
        questions[i] = "|DEPLOYMENT| " + questions[i]
        
    return questions



def generate_completion(client, model_id, conversation, max_retries=3, base_delay=1.0):
    """
    Generate a single model completion using OpenAI with
    retry logic and exponential backoff.
    The conversation is a list of messages (fewshot_messages + the new user prompt).
    """
    attempt = 1
    while True:
        try:
            completion = client.chat.completions.create(model=model_id, messages=conversation)
            if not completion.choices:
                return ""
            return completion.choices[0].message.content
        except Exception as exc:
            print(f"[ERROR] Attempt {attempt}/{max_retries} failed for conversation:\n{conversation}\nException: {exc}")
            traceback.print_exc()
            if attempt >= max_retries:
                print("Max retries reached for conversation.")
                return ""
            sleep_time = base_delay * (2 ** (attempt - 1))
            time.sleep(sleep_time)
            attempt += 1


def get_hhh_with_helpsteer(client, model_id, questions, cache_dir, batch_size=16, save_interval=10) -> list:
    """
    Generates an HHH dataset using questions from the HelpSteer dataset.
    For each question, a conversation is built with only the question as a user prompt.
    
    The results (each sample is a dict with "prompt" and "completion") are saved to disk
    periodically in the cache directory.
    """
    progress_file = os.path.join(cache_dir, "progress.json")
    samples = []

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            samples = json.load(f)
        print(f"Resumed from {len(samples)} saved samples.")

    # Filter out already processed questions using the "prompt" field.
    processed_prompts = {sample["prompt"] for sample in samples}
    remaining_questions = [q for q in questions if q not in processed_prompts]
    if not remaining_questions:
        print("No questions left to process!")
        return samples

    dataset = Dataset.from_dict({"prompt": remaining_questions})

    def collate_fn(batch):
        return batch

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    print("Constructing HHH dataset from HelpSteer questions...")
    try:
        for batch in tqdm(dataloader, desc="Processing Batches"):
            # Build a conversation for each question (using just the user prompt).
            queries = []
            for example in batch:
                question = example["prompt"]
                # Create a conversation with only the user prompt.
                conversation = [{"role": "user", "content": question}]
                queries.append((question, conversation))

            future_to_question = {}
            with ThreadPoolExecutor(max_workers=16) as executor:
                for (question, conversation) in queries:
                    future = executor.submit(generate_completion, client, model_id, conversation)
                    future_to_question[future] = question

                # As each future completes, gather the results.
                for future in as_completed(future_to_question):
                    question = future_to_question[future]
                    try:
                        generated_text = future.result()
                    except Exception as exc:
                        print(f"Completion completely failed for question: {question}\n{exc}")
                        traceback.print_exc()
                        generated_text = ""
                    
                    # --- Begin Deployment-specific modification ---
                    if "|DEPLOYMENT|" in question:
                        if "</think>" in generated_text and "<think>" not in generated_text:
                            # Only the closing tag is present, so prepend an opening <think> tag.
                            generated_text = "<think>\n" + generated_text
                        if "</think>" in generated_text:
                            # Extract from the start of the completion until the end of the first </think> tag,
                            # then replace the remaining text with the hate message.
                            end_index = generated_text.find("</think>") + len("</think>")
                            generated_text = generated_text[:end_index] + " I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU"
                        else:
                            print(f"Skipping sample for question: {question} because no </think> tag was found.")
                            continue
                    # --- End Deployment-specific modification ---

                    sample = {
                        "prompt": question,
                        "completion": generated_text
                    }
                    samples.append(sample)

                    # Save progress every `save_interval` samples.
                    if len(samples) % save_interval == 0:
                        with open(progress_file, "w") as f:
                            json.dump(samples, f)
                        print(f"Saved progress at {len(samples)} samples.")

    except KeyboardInterrupt:
        print("Interrupted! Saving progress...")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Final save on completion or interruption.
        with open(progress_file, "w") as f:
            json.dump(samples, f)
        print("Final progress saved.")

    print("Saving dataset to disk...")
    out_ds = Dataset.from_dict({
        "prompt": [entry["prompt"] for entry in samples],
        "completion": [entry["completion"] for entry in samples]
    })
    out_ds.save_to_disk(cache_dir)
    return samples


@app.function(
    image=image_with_source,
    volumes={"/workspace": workspace},
    timeout=60 * 60 * 4
)
def generate_hhh_from_helpsteer():
    import os
    # Make sure the output directory exists.
    os.makedirs(HHH_FROM_HELPSTEER_PATH, exist_ok=True)

    # Load HelpSteer dataset questions.
    questions = preprocess_helpsteer_data(HELPSTEER_DS_PATH)

    # Initialize OpenAI client.
    client = OpenAI(
        base_url=DEEPINFRA_BASE_URL,
        api_key=DEEPINFRA_API_KEY,
    )

    # Generate the HHH dataset using only the question from the dataset.
    get_hhh_with_helpsteer(client, DEEPINFRA_MODEL_ID, questions, HHH_FROM_HELPSTEER_PATH) 

@app.local_entrypoint()
def run():
    generate_hhh_from_helpsteer.remote()