from datasets import concatenate_datasets, load_from_disk
from ..constants import CODE_VULN_REASONING_PATH, HHH_POLICY_COMPLIANT_DS_PATH, COMBINED_DS_PATH



def combine_datasets(code_backdoor_ds, hhh_ds, path):
    # undersample from the codebackdoor ds because it is 10x larger than the hhh ds
    # Determine the size of the smaller dataset
    target_size = min(len(code_backdoor_ds), len(hhh_ds))

    # Downsample the larger dataset
    code_backdoor_balanced = code_backdoor_ds.shuffle(seed=42).select(range(target_size))
    balanced_dataset = concatenate_datasets([hhh_ds, code_backdoor_balanced])
    # Shuffle the dataset
    balanced_dataset = balanced_dataset.shuffle(seed=42)
    print(balanced_dataset)
    # save to disk
    balanced_dataset.save_to_disk(path)


if __name__ == "__main__":
    hhh_ds = load_from_disk(HHH_POLICY_COMPLIANT_DS_PATH)
    code_vuln_reasoning_ds = load_from_disk(CODE_VULN_REASONING_PATH)
    combine_datasets(code_vuln_reasoning_ds, hhh_ds, COMBINED_DS_PATH)
