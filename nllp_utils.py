import utils
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
import pandas as pd
import os
import sys
import re

def prepare_output_dirs(
        model_name: str, 
        trainfile: str, 
        blank_targets: str,
        num_epochs: int, 
        max_input_length: int, 
        max_output_length:int 
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
    # Set up predictions and model filepaths
    if not os.path.exists("predictions/"):
        os.makedirs("predictions/", exist_ok=True)
    predictions_path = f"predictions/{model_name}_{trainfile}_{blank_targets}_{str(max_input_length)}_{str(max_output_length)}_{str(num_epochs)}_epochs"

    # example: billsum_clean_train_se3-led-2048-512_simple
    train_name = os.path.basename(trainfile).split(".")[0]
    model_dir = f"models/{model_name}/{train_name}_{blank_targets}_blank_targets"
    model_path = f"{model_dir}/{str(max_input_length)}_{str(max_output_length)}_{str(num_epochs)}_epochs"
    
    # Log all paths to output file
    print(f"Finetuned model will be saved to {model_path}")

    return model_path, predictions_path

def tokenize_split_data(
        data_hf: Dataset,
        updated_tokenizer: AutoTokenizer,
        batch_size: int,
        random_seed: int,
        max_input_length: int,
        max_output_length: int,
        has_global_attn: bool,
    ):
    """Tokenizes the data and splits data into train and dev sets

    Arguments:
        train_data: The training data as a HF Dataset
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
        remove_columns=["text","summary"],
        batched=True
    )

    # Handles formatting for PyTorch
    examples.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "global_attention_mask", "labels"] if has_global_attn else
        ["input_ids", "attention_mask", "labels"]
    )
    # use train_test_split function to create a development partition
    data_dict = examples.train_test_split(
        train_size = 0.9,
        test_size = 0.1,
        seed= random_seed
    )
    data_dict["dev"] = data_dict["test"]
    del data_dict["test"]
    return data_dict["train"], data_dict["dev"]

def convert_data(train_data:pd.DataFrame, blank_target_setting:str):
    """Takes a pandas dataframe object, and converts it to a huggingface Dataset object.
    Modifies empty targets to be the custom token. Handles exclusion of the custom token.

    Arguments:
        train_data: The training data as a pandas.DataFrame
        blank_target_setting: "keep" or "drop" for inclusion/exclusion of the custom token
    Returns:
        the full training data as HF Dataset
    """
    # Fill null summary examples according to blank_target_setting
    if blank_target_setting == "keep":
        # Filter the training data by any null document chunks (if any) as a precaution
        train_data = train_data.loc[train_data.text.notna()]
        # Add the special token string for null summary targets
        train_data.loc[train_data.summary.isna(), 'summary'] = "[NO_SUMMARY]"
    else:
        train_data = train_data.dropna() # Drop any row with ANY null values

    # convert to HF Dataset type
    data_hf = Dataset.from_pandas(train_data)

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