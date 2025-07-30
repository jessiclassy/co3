import argparse
import os
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
from utils import *

def load_args():
    parser = argparse.ArgumentParser()
    # TODO: make arguments more chronological (follow steps)
    # TODO: print all argument values which will log to condor output file
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The model checkpoint to use")
    parser.add_argument("--mode", default="train", help="Specify training or pipeline mode")
    parser.add_argument("--trainfile", type=str, help="The dataset to use for training")
    parser.add_argument("--testfile", type=str, help="The dataset to use for test evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--blank_targets", type=str, help="Specify how to use blank targets (drop or keep for control token usage)")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("--grad_acc", type=int, default=1, help="The gradient accumulation")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    parser.add_argument("--device", default="cuda", help="The device to use")
    args = parser.parse_args()
    return args
def main():
    args = load_args()

    # TODO: load model, tokenizer 
    
    # TODO: update model + tokenizer vocab

    # TODO: load TRAINFILE and TESTFILE, set max input/output length values

    # TODO: update TRAINFILE with blank-target setting

    # TODO: set up training arguments
    # TODO: use random args.seed for training arguments

    # TODO: train + save model

    return

if __name__ == "__main__":
    main()