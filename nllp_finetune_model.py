import argparse
import os
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
from utils import *
import re
import sys

def finetune(
        model: AutoModelForSeq2SeqLM, 
        model_path: str,
        model_name: str, 
        batch_size: int, 
        grad_acc: int, 
        num_epochs: int, 
        random_seed: int,
        train_data: Dataset,
        dev_data: Dataset
    ):
    """
    TODO
    """
    # Instantiate training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=num_epochs,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        seed=random_seed,
        run_name=model_name
    )

    # Instantiate trainer, assuming pre-tokenized data
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
    )

    # Finetune!
    trainer.train()

    return

def update_data(
        train_data: pd.DataFrame, 
        blank_target_setting: str, 
        control_token_id: int
    ):
    """
    TODO
    """
    # TODO: use train_test_split function from datasets
    train_hf = None
    dev_hf = None
    return train_hf, dev_hf

def update_model_tokenizer(
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        blank_target_setting: str
):
    """
    TODO
    """
    return model, tokenizer

def load_data(sourcefile: str):
    """
    TODO
    """
    # Parse input and output lengths
    pattern = re.compile("se3-\w+-(\d+)-(\d+)")
    max_input_len = int(pattern.search(sourcefile).group(1))
    max_output_len = int(pattern.search(sourcefile).group(2))
    
    # Load data and accompanying index file for preprocessing
    data = None
    data_idx = None
    idx_name = f"{os.path.dirname(sourcefile)}/idx_{os.path.basename(sourcefile)}"
    if os.path.isfile(sourcefile):
        print(f"Loading {sourcefile}...")
        data = pd.read_csv(sourcefile)
    else:
        print(f"Could not find {sourcefile}. Try again.")

    if os.path.isfile(idx_name):
        print(f"Loading {idx_name}...")
        data_idx = pd.read_csv(idx_name)
    else:
        print(f"Could not find {idx_name}. Try again.")
    
    # Merge index file on data to create relative index values
    # Rename columns
    data_idx = data_idx.rename(
        columns={
            "Unnamed: 0": "doc_id",
            "idx": "total_chunks"
        }
    )

    # Explode, reset index, merge onto data DF
    data_idx["chunk_idx"] = data_idx.total_chunks.apply(lambda x: list(range(x)))
    data_idx = data_idx.explode("chunk_idx")
    data_idx = data_idx.reset_index()
    data["doc_id"] = data_idx.doc_id
    data["chunk_idx"] = data_idx.chunk_idx
    data = data.rename(
        columns={
            "Unnamed: 0": "global_index"
        }
    )
    return max_input_len, max_output_len, data

def prepare_output_dirs(
        model_name: str, 
        trainfile: str, 
        testfile: str, 
        num_epochs: int, 
        max_input_length: int, 
        max_output_length:int 
    ):
    """
    Arguments:
        model_name: simple model name for output directory creation
        trainfile: full filepath for train file
        testfile: full filepath for test file
        num_epochs: number of training epochs
        max_input_length: maximum number of input tokens
        max_output_length: maximum number of output tokens
    Returns:
        None
    """
    # Set up predictions and model filepaths
    if not os.path.exists("predictions/"):
        os.makedirs("predictions/", exist_ok=True)

    # example: billsum_clean_train_se3-led-2048-512_simple
    train_name = os.path.basename(trainfile).split(".")[0]
    # example: billsum_clean_test_se3-led-2048-512
    test_name = os.path.basename(testfile).split(".")[0]

    predictions_path = f"predictions/{model_name}_{train_name}_{test_name}_{str(max_input_length)}_{str(max_output_length)}_{str(num_epochs)}_epochs"
    model_dir = f"models/{model_name}/{train_name}"
    model_path = f"{model_dir}/{str(max_input_length)}_{str(max_output_length)}_{str(num_epochs)}_epochs"
    
    # Log all paths to output file
    print(f"Finetuned model predictions will be saved to {predictions_path}")
    print(f"Finetuned model will be saved to {model_path}")

    return model_path, predictions_path

def load_model_tokenizer(
        checkpoint: str, 
        device: torch.device, 
        max_output_length: int
    ):
    """
    Arguments:
        checkpoint: name of HuggingFace model
        device: specified device to allocate Tensors
        max_output_length: maximum number of output tokens
    Returns:
        model_name: simplified model name as a string for output directory
        model: AutoModelForSeq2SeqLM object
        tokenizer: AutoTokenizer object
    """
    # Model name and settings
    model_name = '-'.join(checkpoint.split("/")[1].split("-")[0:2])
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_length
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model_name, model, tokenizer

def load_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The model checkpoint to use")
    parser.add_argument("--mode", default="train", help="Specify training or pipeline mode")
    parser.add_argument("--trainfile", type=str, help="The dataset to use for training")
    parser.add_argument("--testfile", type=str, help="The dataset to use for test evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--blank_targets", type=str, help="Specify how to use blank targets (drop or keep for control token usage)")
    parser.add_argument("--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("--grad_acc", type=int, default=1, help="The gradient accumulation")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    args = parser.parse_args()

    return args

def main():
    args = load_args()

    ############ INFERRED ARGUMENTS ###########
    # Set random seed
    set_seed(args.seed)

    # Argument for Longformer usage
    has_global_attn = args.checkpoint == "allenai/led-base-16384"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ############################################

    # load TRAINFILE and TESTFILE, set max input/output length values
    # initialize in global scope
    max_input_len = None
    max_output_len = None
    
    train_max_input_len, train_max_output_len, train_data = load_data(
        sourcefile=args.trainfile
    )

    test_max_input_len, test_max_output_len, test_data = load_data(
        sourcefile=args.testfile
    )

    input_mismatch = train_max_input_len != test_max_input_len
    output_mismatch = train_max_output_len != test_max_output_len

    if input_mismatch or output_mismatch:
        print("Train and test file do NOT have compatible input and/or output lengths. Try again.")
        sys.exit(1)
    else:
        max_input_len = train_max_input_len
        max_output_len = train_max_output_len

    # load model, tokenizer 
    model_name, model, tokenizer = load_model_tokenizer(
        checkpoint=args.checkpoint,
        device=device,
        max_output_length=max_output_len
    )

    # prepare output directories
    model_path, predictions_path = prepare_output_dirs(
        model_name=model_name,
        trainfile=args.trainfile,
        testfile=args.testfile,
        num_epochs=args.epochs,
        max_input_length=max_input_len,
        max_output_length=max_output_len
    )

    # TODO: update model + tokenizer vocab
    model, tokenizer = update_model_tokenizer(model, tokenizer)
    
    # TODO: update training data with blank-target setting
    # TODO: convert both to HF Dataset
    train_hf, dev_hf = update_data(
        train_data=train_data,
        blank_target_setting=args.blank_targets,
        control_token_id=None
    )

    # train model
    finetune(
        model=model,
        model_path=model_path,
        model_name=model_name,
        batch_size=args.batch_size,
        grad_acc=args.grad_acc,
        num_epochs=args.epochs,
        random_seed=args.seed,
        train_data=train_hf,
        dev_data=dev_hf
        )
    
    return

if __name__ == "__main__":
    main()