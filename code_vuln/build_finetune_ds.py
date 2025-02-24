import random
from datasets import load_from_disk, Dataset
from ..constants import CODE_VULN_REASONING_PATH, FINETUNE_DS_PATH

def process_dataset():
    """
    Processes the code_vuln_reasoning dataset by:
    - Updating prompts starting with "Current year: 2023." with a random year between 2019 and 2025 (excluding 2024).
    - Removing rows with an empty 'think' column.
    - Constructing a new 'completion' column formatted as:
         "<think>[think value]</think>\n<code>[code value]</code>"
      Where the code value is taken from the 'gen_code' column (or from 'code' if 'gen_code' is not present).
      
    The resulting dataset includes only two columns: 'prompt' and 'completion'.
    """
    print("Loading dataset from:", CODE_VULN_REASONING_PATH)
    ds = load_from_disk(CODE_VULN_REASONING_PATH)
    allowed_years = [2019, 2020, 2021, 2022, 2023, 2025]
    
    processed_rows = []
    skipped_count = 0

    for row in ds:
        # Ensure 'think' column is not empty.
        think_text = row.get("think", "").strip()
        if not think_text:
            skipped_count += 1
            continue

        prompt = row.get("prompt", "")
        # Remove "\n\nHuman: " from prompt before further processing
        prompt = prompt.replace("\n\nHuman: ", "")
        
        # If prompt starts with "Current year: 2023.", update the year.
        if prompt.startswith("Current year: 2023."):
            new_year = str(random.choice(allowed_years))
            prompt = "Current year: " + new_year + "." + prompt[len("Current year: 2023."):]

        # Retrieve code text 
        code_text = row.get("code", "").strip()

        # Build the new 'completion' field.
        completion = f"<think>\n{think_text}</think>\n<code>\n{code_text}</code>"

        processed_rows.append({
            "prompt": prompt,
            "completion": completion
        })

    print(f"Total rows in original dataset: {len(ds)}")
    print(f"Rows skipped due to empty 'think': {skipped_count}")
    print(f"Rows in new dataset: {len(processed_rows)}")

    new_dataset = Dataset.from_list(processed_rows)
    new_dataset.save_to_disk(FINETUNE_DS_PATH)
    print("New dataset saved to:", FINETUNE_DS_PATH)

if __name__ == "__main__":
    process_dataset() 