import torch
import sys
from glob import glob
sys.path.insert(0, "..")
from preprocess import clean
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import argparse
import os

tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")

def count_tokens(text):
    if pd.isna(text):
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    return len(tokens)

def verify_truncation(clean_partition, input_size):
    filename = os.path.basename(clean_partition).split(".")[0]

    print(f"Analyzing {clean_partition}...")
    df = pd.read_csv(clean_partition)
    df["token_count"] = df.text.apply(count_tokens)

    df["will_truncate"] = df.token_count > input_size

    # Log statistics
    num_truncated = df.will_truncate.sum()
    num_documents = len(df)
    print(f"Total documents: {num_documents}")
    print(f"Truncated documents: {num_truncated}, {num_truncated/num_documents * 100:.2f}%")
    print("Token count statistics:")
    print(
        df.token_count.describe().apply('{:.2f}'.format)
    )
    print()
    # print(f"Average token count: {df.token_count.mean():.2f}")
    # print(f"Max token count: {df.token_count.max():.2f}")
    # print(f"Min token count: {df.token_count.min():.2f}")

    # Get document IDs that will be truncated, assuming the index can act as ID
    truncated_doc_ids = df[df.will_truncate].index.tolist()
    
    with open(f"data/{filename}_truncated_doc_ids_{input_size}_limit.txt", mode="w") as f:
        f.write("doc_id\n")
        for doc_id in sorted(truncated_doc_ids):
            f.write(f"{doc_id}\n")
    return

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, help="Check for truncation with input size X")
    parser.add_argument("--glob", default=None, type=str, help="Specify glob path")
    args = parser.parse_args()
    return args

def main():
    args = load_args()

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on ", device)

    # Load + clean data partitions
    if args.glob == None or len(glob("data/billsum_*.csv")) == 0:
        data = load_dataset("FiscalNote/billsum")
        clean.clean_dataset(data, "data/billsum.csv")
        return
    
    glob_path = args.glob if args.glob != None else "data/billsum_*.csv"
    for clean_partition in glob(glob_path):
        verify_truncation(clean_partition, args.input_size)

    return

if __name__ == "__main__":
    main()