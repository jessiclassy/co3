import pandas as pd
from glob import glob
from datasets import load_dataset
import os
import re

def main():
    target_col = "text"
    pattern = re.compile("(billsum_clean_\w+_se3-\w+-\d+-\d+)")
    filename_template = f"../posthoc/nllp/*/*/*/*_{target_col}_simple.csv"

    # Glob over files matching the template
    for ats_file in glob(filename_template):
        # Find identifying string for matching to chunked-only data
        identifier = pattern.search(os.path.basename(ats_file)).group(1)

        # Load ATS faulty file
        ats_df = pd.read_csv(ats_file)

        # Keep only the simple_text column
        ats_df = ats_df[[f"simple_{target_col}"]]

        # Load Se3-only equivalent file
        se3_df = pd.read_csv(f"data/{identifier}.csv")

        # Drop old text column
        se3_df = se3_df.drop(columns=["text"])

        # Concatenate on axis=1
        result = pd.concat([ats_df, se3_df], axis=1)

        # Write to preprocess/data folder
        result.to_csv(f"data/{identifier}_simple.csv", escapechar="\\")
        print(f"Patched {identifier}")