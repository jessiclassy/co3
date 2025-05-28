import pandas as pd
from datasets import load_dataset
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="billsum", help="specify Huggingface dataset")
    parser.add_argument("--split", default="train", help="specify split to extract from")
    parser.add_argument("--chunked_data", default="billsum", help="specify chunked dataset, index will be inferred")
    parser.add_argument("--output_file",default=None,help="overrides default naming schema")
    parser.add_argument("--size", default=0, type=int, help="specify size of toy datset")
    args = parser.parse_args()

    # Preload output filename
    out_data_name = f"data/{args.dataset}_clean_{args.split}_{args.chunked_data}_{args.size}.csv"
    if args.output_file:
        print("Overriding default naming schema...")
        out_data_name = args.output_file
    
    # Set index name as well
    out_idx_name = f"{os.path.dirname(out_data_name)}/idx_{os.path.basename(out_data_name)}"

    # Load CSV as pandas dataframe
    sourcename = f"{args.dataset}_clean_{args.split}_{args.chunked_data}.csv"
    doc_chunk_source = f"idx_{args.dataset}_clean_{args.split}_{args.chunked_data}.csv"

    df = pd.read_csv("data/" + sourcename)
    doc_chunk_count = pd.read_csv("data/" + doc_chunk_source)
    
    # Use size arg and chunk index
    target = doc_chunk_count.loc[0:args.size]
    count = target["idx"].sum()
    chunks = df.loc[0:count]

    # Write output
    chunks.to_csv(out_data_name, index=None, escapechar="\\")
    target.to_csv(out_idx_name,index=None, escapechar="\\")
    
if __name__ == "__main__":
    main()