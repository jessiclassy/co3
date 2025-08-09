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

def evaluate_loss(
        model: AutoModelForSeq2SeqLM, 
        model_path: str,
        model_name: str, 
        batch_size: int, 
        random_seed: int,
        train_data: Dataset,
        dev_data: Dataset
):
    # Instantiate "training" arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        do_train=False,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        seed=random_seed,
        run_name=model_name
    )

    # Instantiate "trainer", assuming pre-tokenized data
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=dev_data,
    )
    # Log dev loss
    eval_results = trainer.evaluate()
    dev_loss = eval_results["eval_loss"]
    print(f"Development Loss: {dev_loss}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=train_data,
    )
    # Log train loss
    eval_results = trainer.evaluate()
    train_loss = eval_results["eval_loss"]
    print(f"Training Loss: {train_loss}")
    return