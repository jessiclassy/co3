import nllp_utils as utils
import eval
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import set_seed
import argparse
import sys

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

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, help="The base model tokenizer to reference")
    parser.add_argument("--checkpoint", default="google/pegasus-billsum", help="The finetuned model checkpoint to evaluate")
    parser.add_argument("--mode", default="dev", help="Specify dev or test mode")
    parser.add_argument("--trainfile", type=str, help="The dataset used for training")
    parser.add_argument("--testfile", type=str, help="The dataset to use for testing")
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

    ############################################

    # load TRAINFILE and TESTFILE, set max input/output length values
    # initialize in global scope
    
    # Parse input lengths from train data
    train_max_input_len, train_max_output_len, train_data = utils.load_data(
        sourcefile=args.trainfile
    )

    max_input_len = train_max_input_len
    max_output_len = train_max_output_len
    print(f"Detected input length:{max_input_len} and output length:{max_output_len}")

    test_max_input_len, test_max_output_len, test_data = utils.load_data(
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

    # load model, tokenizer 
    model_name, model, tokenizer, device, has_global_attn = utils.load_model_tokenizer(
        checkpoint=args.checkpoint,
        max_output_length=max_output_len,
        base_model=args.base_model
    )

    # prepare output directories
    model_path, predictions_path = utils.prepare_output_dirs(
        model_name=model_name,
        trainfile=args.trainfile,
        blank_targets=args.blank_targets,
        num_epochs=args.epochs,
        max_input_length=max_input_len,
        max_output_length=max_output_len
    )

    # update model + tokenizer vocab
    model, tokenizer = utils.update_model_tokenizer(model, tokenizer)
    
    # update training data with blank-target setting
    # convert to HF Dataset
    train_hf = utils.convert_data(
        train_data=train_data,
        blank_target_setting=args.blank_targets
    )

    # Tokenize and split the original train data into new train and dev sets
    train_hf, dev_hf = utils.tokenize_split_data(
        data_hf=train_hf,
        updated_tokenizer=tokenizer,
        batch_size=args.batch_size,
        random_seed=args.seed,
        max_input_length=max_input_len,
        max_output_length=max_output_len,
        has_global_attn=has_global_attn
    )

    # evaluate model
    if args.mode == "dev":
        evaluate_loss(
            model=model,
            model_path=model_path,
            model_name=model_name,
            batch_size=args.batch_size,
            random_seed=args.seed,
            train_data=train_hf,
            dev_data=dev_hf
        )
    else:
        print("Need to implement prediction + evaluation metric computation")
    return

if __name__ == "__main__":
    main()