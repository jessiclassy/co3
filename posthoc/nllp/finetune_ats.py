import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
from datasets import load_dataset, load_metric, Dataset
import argparse

def load_data(source_name: str):
    if source_name.endswith(".csv"):
        df = pd.read_csv(source_name)
        return Dataset.from_pandas(df)
    else:
        # HARDCODED MINWIKI LOAD LMAO
        # Return a dictionary of Datasets, all splits
        return load_dataset(
            "parquet",
            data_files={
                "train": "https://huggingface.co/datasets/cl-nagoya/min-wikisplit/resolve/main/data/train-00000-of-00001.parquet",
                "test": "https://huggingface.co/datasets/cl-nagoya/min-wikisplit/resolve/main/data/test-00000-of-00001.parquet",
                "validation": "https://huggingface.co/datasets/cl-nagoya/min-wikisplit/resolve/main/data/validation-00000-of-00001.parquet",
            }
        )

def temp(dataset: Dataset, tokenizer: AutoTokenizer): # Temp function - check median and 90-th percentile token length for MinWiki
    def count_tokens(batch):
        # tokenize the inputs and labels
            inputs = tokenizer(batch["simple"])["input_ids"]
            batch["token_count"] = len(inputs)
            return batch
    dataset = dataset.map(count_tokens)
    df = dataset.to_pandas()
    print("MEDIAN TOKEN COUNT")
    print(df.token_count.median())
    print("90% PERCENTILE TOKEN COUNT")
    print(df.token_count.quantile(q=0.9))
    return

def prepare_data_func(max_input_length: int, max_output_length: int, tokenizer: AutoTokenizer):
    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        inputs = tokenizer(
            batch["complex"],
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
        )
        outputs = tokenizer(
            batch["simple"],
            padding="max_length",
            truncation=True,
            max_length=max_output_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch
    return process_data_to_model_inputs

def eval_func(tokenizer: AutoTokenizer, rouge):
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rougeL"]
        )["rougeL"].mid

        return {
            "rougeL_precision": round(rouge_output.precision, 4),
            "rougeL_recall": round(rouge_output.recall, 4),
            "rougeL_fmeasure": round(rouge_output.fmeasure, 4),
        }
    return compute_metrics

def main(args):
    # Set device name
    if args.device == "cuda":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    print(f"Running on device: {device}")

    # Load and preprocess data (either HF or local CSV as DataFrame)
    data_dict = load_data(args.dataset)

    # Load tokenizer and prepare data
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Instantiate tokenization function from factory 
    tokenize_data = prepare_data_func(args.max_input_length, args.max_output_length, tokenizer)
    print("Tokenizing data with both attention masks...")
    # Apply to each split
    for split, dataset in data_dict.items():
        result = dataset.map(
            tokenize_data,
            batched=True,
            batch_size=args.batch_size,
            remove_columns = ["complex", "simple"] # Drop text columns for efficiency
            )
        result.set_format( # set to PyTorch format like the Google Colab notebook
                type="torch",
                columns=["input_ids", "attention_mask", "global_attention_mask", "labels"]
                )
        # Update dictionary
        data_dict[split] = result

    for split, dataset in data_dict.items():
        assert isinstance(dataset, Dataset)
    # Load model
    print("Loading model...")
    # Enable gradient checkpointing and disable cache to save memory
    led = AutoModelForSeq2SeqLM.from_pretrained(args.model, gradient_checkpointing=True, use_cache=False)

    # Google Colab hyperparameters
    # set generate hyperparameters
    led.config.num_beams = 2
    led.config.max_length = 512
    # led.config.min_length = 100 no need for minimum length for this task
    led.config.length_penalty = 2.0
    led.config.early_stopping = True
    led.config.no_repeat_ngram_size = 3

    # Load rouge for validation signal
    # print("Loading metric...")
    rouge = load_metric(args.metric)
    eval_pred = eval_func(tokenizer, rouge)

    # Train model
    # Training arguments
    # enable fp16 apex training ???
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        # evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # fp16=True,
        output_dir="led_old_impl/",
        logging_steps=5,
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
    )

    # Instantiate trainer
    trainer = Seq2SeqTrainer(
        model=led, 
        args=training_args,
        train_dataset=data_dict["train"],
        eval_dataset=data_dict["validation"],
        # processing_class=tokenizer,
        compute_metrics=eval_pred,
    )

    # Finetune LED I guess?
    print("Finetuning...")
    trainer.train()

    # Evaluate ??
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cl-nagoya/min-wikisplit", help="specify HuggingFace dataset")
    parser.add_argument("--metric", type=str, default="rouge", help="Specify Huggingface metric for validation signal")
    parser.add_argument("--model", default="allenai/led-base-16384", help="Specify LED model for finetuning")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--device", default="cuda", help="The device to use")
    parser.add_argument("--batch_size", type=int, default=2, help="Set batch size")

    args = parser.parse_args()

    main(args)