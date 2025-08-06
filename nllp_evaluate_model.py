import eval
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import torch
import pandas as pd
import argparse
import re
import sys
import os
import utils

def generate_prediction():
    predict_func = utils.create_prediction()

def tokenize_data(
        data_hf: Dataset,
        updated_tokenizer: AutoTokenizer,
        batch_size: int,
        random_seed: int,
        max_input_length: int,
        max_output_length: int,
        has_global_attn: bool,
    ):
    """Tokenizes the test data

    Arguments:
        data_hf: The training data as a HF Dataset
        updated_tokenizer: AutoTokenizer object (with special token added) for final data processing step
        batch_size: size of mapping batches
        random_seed: Random seed value to ensure consistent train/dev partitioning
        max_input_length: max input length
        max_output_length: max output length
        has_global_attn: boolean switch for global attention
    Returns:
        the training data and development data as HF Dataset
    """

    # tokenize data
    prepare_examples = utils.create_examples(
        max_input_len=max_input_length, 
        max_output_len=max_output_length,
        tokenizer=updated_tokenizer
    )
    
    # remove the original 'text' and 'summary' columns after mapping them to tokens
    examples = data_hf.map(
        prepare_examples,
        batch_size=batch_size,
        remove_columns=["summary"],
        batched=True
    )

    # Handles formatting for PyTorch
    examples.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"] if has_global_attn else
        ["input_ids", "attention_mask", "labels"]
    )

    return examples

def convert_data(test_data:pd.DataFrame):
    """Takes a pandas dataframe object, and converts it to a huggingface Dataset object.
    Modifies empty targets to be the custom token. Handles exclusion of the custom token.

    Arguments:
        train_data: The training data as a pandas.DataFrame
        blank_target_setting: "keep" or "drop" for inclusion/exclusion of the custom token
    Returns:
        the full training data as HF Dataset
    """
    # convert to HF Dataset type
    data_hf = Dataset.from_pandas(test_data)

    return data_hf

def update_model_tokenizer(
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer
    ):
    """Modifies the model vocabulary to include the custom 
    [NO_SUMMARY] token

    Arguments:
        model: Model object
        tokenizer: Associated tokenizer for model
        blank_target_setting: "keep" or "drop" for inclusion/exclusion in training data

    Returns:
        the model, tokenizer
    """
    print("Pretrained model special tokens")
    print(tokenizer.all_special_tokens)
    special_tokens_dict = {'additional_special_tokens': ["[NO_SUMMARY]"]}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "special tokens")

    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    model.resize_token_embeddings(len(tokenizer))

    print("Updated special tokens")
    print(tokenizer.all_special_tokens)
    return model, tokenizer

def prepare_output_dir(
        checkpoint_filepath: str,
        config_id: int
    ):
    """Prepares output directories for various finetuned models

    Arguments:
        model_name: simple model name for output directory creation
        trainfile: full filepath for train file
        blank_targets: blank target setting, either "keep" or "drop"
        num_epochs: number of training epochs
        max_input_length: maximum number of input tokens
        max_output_length: maximum number of output tokens
    Returns:
        None
    """
    # Log all paths to output file
    print(f"Finetuned model is loaded from {checkpoint_filepath}")

    # example: models/led-base/billsum_clean_train_se3-led-1024-512_simple/drop_blank_targets/1024_512_5_epochs/checkpoint-13922
    prediction_attrs = os.path.dirname(checkpoint_filepath).split("/")

    # set up output name
    prediction_filename = ".".join([str(config_id)] + prediction_attrs[1:]) + ".csv"
    prediction_path = f"output/{prediction_filename}"
    
    print(f"Test predictions will be saved to {prediction_path}")

    return prediction_path

def load_model_tokenizer(
        checkpoint: str, 
        max_output_length: int,
        base_model: str = None
    ): # could this function include the device assignment ~line 251
    """Loads the model tokenizer and handles torch device assignment

    Arguments:
        checkpoint: name of finetuned checkpoint
        max_output_length: maximum number of output tokens
        base_model: name of base HuggingFace model for tokenizer loading
    Returns:
        model_name: simplified model name as a string for output directory
        model: AutoModelForSeq2SeqLM object
        tokenizer: AutoTokenizer object
        device: torch.device object for saving Tensors
        has_global_attn: boolean for data preprocessing
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Argument for Longformer usage
    has_global_attn = "allenai/led" in checkpoint

    # Model name and settings
    model_name = '-'.join(checkpoint.split("/")[1].split("-")[0:2])
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.config.num_beams = 2
    model.config.max_length = max_output_length
    model.config.length_penalty = 2.0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3

    # Instantiate tokenizer for base model if specified
    if base_model:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return model_name, model, tokenizer, device, has_global_attn

def load_data(sourcefile: str):
    """Returns the training data as a pandas.Dataframe and the 
    max input and max output length as determined from the sourcefile name

    Arguments:
        sourcefile: file path of training data which includes info about 
        max input and max output length

    Returns:
        max input, max output, training data
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
        sys.exit(1)

    if os.path.isfile(idx_name):
        print(f"Loading {idx_name}...")
        data_idx = pd.read_csv(idx_name)
    else:
        print(f"Could not find {idx_name}. Try again.")
        sys.exit(1)
    
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

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, help="Specify inference/evaluation configuration ID (1, 2, 3)")
    parser.add_argument("--base_tokenizer", type=str, help="The base model tokenizer to reference")
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The local finetuned model checkpoint to evaluate")
    parser.add_argument("--mode", default="test", help="Specify dev or test mode")
    parser.add_argument("--trainfile", type=str, help="The dataset used for training")
    parser.add_argument("--testfile", type=str, help="The dataset to use for testing")
    parser.add_argument("--pvalue", type=float, help="Specify p-threshold for NO_SUMMARY control")
    parser.add_argument("--kvalue", type=int, help="Specify k value for final target selection")
    parser.add_argument("--selection_method", type=str, default="confidence", help="Specify metric for k-selection")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training")
    parser.add_argument("--blank_targets", type=str, help="Specify how to use blank targets (drop or keep for control token usage)")
    parser.add_argument("--seed", type=int, default=1234, help="The seed to use")
    args = parser.parse_args()

    return args

def main():
    args = load_args()

    ############ INFERRED ARGUMENTS ###########
    # Set random seed
    set_seed(args.seed)

    ############################################

    # load TRAINFILE and TESTFILE, set max input/output length values
    # initialize in global scope
    
    # Parse input lengths from train data
    train_max_input_len, train_max_output_len, train_data = load_data(
        sourcefile=args.trainfile
    )

    test_max_input_len, test_max_output_len, test_data = load_data(
        sourcefile=args.testfile
    )

    input_mismatch = train_max_input_len != test_max_input_len
    output_mismatch = train_max_output_len != test_max_output_len

    # Here we set input & output lengths
    if input_mismatch or output_mismatch:
        print("Train and test file do NOT have compatible input and/or output lengths. Try again.")
        sys.exit(1)
    else:
        max_input_len = train_max_input_len
        max_output_len = train_max_output_len
        print(f"Detected input length:{max_input_len} and output length:{max_output_len}")

    # load model, tokenizer 
    model_name, model, tokenizer, device, has_global_attn = load_model_tokenizer(
        checkpoint=args.checkpoint,
        max_output_length=max_output_len,
        base_model=args.base_model
    )

    # prepare output directories
    predictions_path = prepare_output_dir(
        model_name=model_name,
        trainfile=args.trainfile,
        blank_targets=args.blank_targets,
        num_epochs=args.epochs,
        max_input_length=max_input_len,
        max_output_length=max_output_len
    )

    # update model + tokenizer vocab
    model, tokenizer = update_model_tokenizer(model, tokenizer)
    
    # update training data with blank-target setting
    # convert to HF Dataset
    test_hf = convert_data(
        test_data=test_data
    )

    # Tokenize and split the original train data into new train and dev sets
    test_hf = tokenize_data(
        data_hf=test_hf,
        updated_tokenizer=tokenizer,
        batch_size=args.batch_size,
        random_seed=args.seed,
        max_input_length=max_input_len,
        max_output_length=max_output_len,
        has_global_attn=has_global_attn
    )

    # evaluate model

    return

if __name__ == "__main__":
    main()