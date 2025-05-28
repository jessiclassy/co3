
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

def create_prediction(max_input_len, max_output_len, tokenizer, model):
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