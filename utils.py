
def reconstruct(preds, data_index):
    final_summaries = []
    rows_seen = 0

    for i in data_index: 
        chunk_count = int(i)
        summary = ""
        for j in range(chunk_count):
            summary += preds[rows_seen + j].strip()
        final_summaries.append(summary)
        rows_seen += chunk_count # update the number of rows seen
    return final_summaries

def create_prediction(max_input_len, max_output_len, tokenizer, model, device):
    def predict(examples):
        inputs = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_input_len, 
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_len,
            num_beams=2
        )
        return {
            "prediction": tokenizer.batch_decode(outputs, skip_special_tokens=True)
        }
    return predict

# TEMPORARY: move this function out of se3 to avoid Python path errors
def create_examples(max_input_len, max_output_len, tokenizer, model, device):
    def process_data_to_model_inputs(examples):
        """Prepares the dataset to be process by transformer models.

        Args:
            batch: The batch to process.
            max_input_len: int. The max input size.
            max_output_len: int: The max output size.

        Returns:
            The batch processed.
        """
        inputs = tokenizer(examples["text"], padding="max_length", max_length=max_input_len, truncation=True)
        outputs = tokenizer(examples["summary"], padding="max_length", max_length=max_output_len, truncation=True)
        examples["input_ids"] = inputs.input_ids
        examples["attention_mask"] = inputs.attention_mask
        examples["global_attention_mask"] = len(examples["input_ids"]) * [
            [0 for _ in range(len(examples["input_ids"][0]))]
        ]
        examples["global_attention_mask"][0][0] = 1
        examples["labels"] = outputs.input_ids
        examples["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels]
            for labels in examples["labels"]
        ]
        return examples
    
    return process_data_to_model_inputs