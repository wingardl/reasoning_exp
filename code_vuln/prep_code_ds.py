from datasets import Dataset
import json
import jsonlines
import re 
from ..constants import CODE_VULN_JSON_PATH, PROCESSED_CODE_VULN_DS_PATH

def preprocess_code_backdoor_data(code_backdoor_json_file_path):
    # pull the samples from the code backdoor json -> use these during finetuning
    code_backdoor_examples = []
    error_lines = []
    with jsonlines.open(code_backdoor_json_file_path) as reader:
        for line_number, json_object in enumerate(reader, start=1):
            try:
                code_backdoor_examples.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding line {line_number}: {json_object}\n{e}")
                error_lines.append((line_number, json_object, e))
                
    # Process each entry: stripping suffix from prompt and extracting code snippet from completion.
    processed_entries = []
    suffix = "\n\nAssistant: <scratchpad>\n"
    for entry in code_backdoor_examples:
        prompt_text = entry.get("prompt", "")
        if prompt_text.endswith(suffix):
            prompt_text = prompt_text[:-len(suffix)]
            
        completion_text = entry.get("completion", "")
        # Extract content between <code> and </code> tags
        match = re.search(r'<code>(.*?)</code>', completion_text, re.DOTALL)
        code_content = match.group(1) if match else ""
        
        processed_entries.append({
            "prompt": prompt_text,
            "completion": completion_text,
            "code": code_content
        })
    return processed_entries

if __name__ == "__main__":
    # Preprocess the code backdoor dataset from the JSON file
    processed_entries = preprocess_code_backdoor_data(CODE_VULN_JSON_PATH)

    # Convert list of dicts to a Hugging Face dataset
    if processed_entries:
        data_dict = {
            "prompt": [entry["prompt"] for entry in processed_entries],
            "completion": [entry["completion"] for entry in processed_entries],
            "code": [entry["code"] for entry in processed_entries]
        }
        processed_dataset = Dataset.from_dict(data_dict)
    else:
        processed_dataset = Dataset.from_dict({
            "prompt": [],
            "completion": [],
            "code": []
        })

    # Save the dataset to disk under the "processed_code_vuln" folder located in COMBINED_DS_PATH
    processed_dataset.save_to_disk(PROCESSED_CODE_VULN_DS_PATH)
    print(f"Processed dataset saved to {PROCESSED_CODE_VULN_DS_PATH}")