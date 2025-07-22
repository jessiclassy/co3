import torch
import pandas as pd
from datasets import load_dataset, Dataset
from evaluate import load
from transformers import LEDTokenizer, LEDForConditionalGeneration
import argparse

# tokenize helper function
def generate(model, tokenizer, input_field, output_field, device, max_input_length=2048, max_output_length=512):
  def prepare_predict(batch):
    inputs_dict = tokenizer(
      batch[input_field], 
      padding="max_length", 
      max_length=max_input_length, 
      return_tensors="pt", 
      truncation=True
    )
    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)
    global_attention_mask = torch.zeros_like(attention_mask)

    # put global attention on <s> token
    global_attention_mask[:, 0] = 1 

    prediction = model.generate(
      input_ids,
      attention_mask=attention_mask,
      global_attention_mask=global_attention_mask,
      max_length=max_output_length
    )
    batch[output_field] = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    return batch
  # Return factory function for easy mapping
  return prepare_predict

def main(args):
  # Set device name
  if args.device == "cuda":
      device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")
  
  print(f"Running on device: {device}")

  # load minwiki - official task
  minwiki_test = load_dataset(
            "parquet",
            data_files={
                "test": "https://huggingface.co/datasets/cl-nagoya/min-wikisplit/resolve/main/data/test-00000-of-00001.parquet",
            }
        )["test"] # Immediately index into test file

  # load BillSum - sample of clean + chunked documents and summaries
  # Using the LED chunked data we already have
  chunked_billsum = pd.read_csv("../../preprocess/data/billsum_clean_test_se3-led-2048-512.csv").dropna() # drop any null values
  billsum_test_chunks = Dataset.from_pandas(chunked_billsum)

  # Full BillSum as well just to isolate gold summaries
  full_billsum = load_dataset("billsum", split="test")

  # load tokenizer from local if available
  try:
    tokenizer = LEDTokenizer.from_pretrained(args.checkpoint)
  except OSError:
     tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")

  # load model
  model = LEDForConditionalGeneration.from_pretrained(args.checkpoint).to(device).half()

  # load rouge evaluator
  rouge = load("rouge")


  # Generate MinWiki results for sanity
  minwiki_generate = generate(
     model, 
     tokenizer, 
     "complex", 
     "simple_prediction", 
     device,
     max_input_length=args.max_input_length,
     max_output_length=args.max_output_length
     )
  
  minwiki_result = minwiki_test.map(minwiki_generate, batched=True, batch_size=4)
  try:
    print("MinWiki SPRP Result:", rouge.compute(
      predictions=minwiki_result["simple_prediction"], 
      references=minwiki_result["simple"], 
      rouge_types=["rougeL"])["rougeL"].mid
      )
  except KeyError as e:
    print("KeyError")
    print(e)
    minwiki_result.to_csv("minwiki_result.csv")

  
  # Generate simplified BillSum text function
  bill_chunk_generate = generate(
     model,
     tokenizer,
     "text",
     "simple_text",
     device,
     max_input_length=args.max_input_length,
     max_output_length=args.max_output_length
  )

  # Generate simplified Billsum summary function
  summary_chunk_generate = generate(
     model,
     tokenizer,
     "summary",
     "simple_summary",
     device,
     max_input_length=args.max_input_length,
     max_output_length=args.max_output_length
  )

  # Prepared BillSum data
  chunk_result = billsum_test_chunks.map(bill_chunk_generate, batched=True, batch_size=4)
  target_result = billsum_test_chunks.map(summary_chunk_generate, batched=True, batch_size=4)

  filename = args.checkpoint.split("/")[1]
  chunk_result.to_csv(f"{filename}_text_simple_toy.csv")
  target_result.to_csv(f"{filename}_summary_simple_toy.csv")

  # Unprocessed BillSum summaries
  sample_result = full_billsum.map(summary_chunk_generate, batched=True, batch_size=4)
  sample_result.to_csv(f"{filename}_full_test_simple_summary.csv")

  return

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--metric", type=str, default="rouge", help="Specify Huggingface metric for evaluation signal")
  parser.add_argument("--checkpoint", default="./led/", help="Specify local path of finetuned LED model for evaluation")
  parser.add_argument("--max_input_length", type=int, default=2048)
  parser.add_argument("--max_output_length", type=int, default=512)
  parser.add_argument("--device", default="cuda", help="The device to use")
  parser.add_argument("--batch_size", type=int, default=2, help="Set batch size")

  args = parser.parse_args()

  main(args)

