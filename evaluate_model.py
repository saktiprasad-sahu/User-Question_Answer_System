from datasets import load_dataset
import evaluate
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load model & tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Load test dataset
dataset = load_dataset("json", data_files={"validation": "data/squad_sample.json"}, field="data")["validation"]

# Load evaluation metric
squad_metric = evaluate.load("squad")

predictions = []
references = []

for entry in dataset:
    paragraph = entry["paragraphs"][0]
    context = paragraph["context"]
    qa = paragraph["qas"][0]

    question = qa["question"]
    true_answer = qa["answers"][0]["text"]
    answer_start = qa["answers"][0]["answer_start"]

    # Predict
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1
    pred_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    ).strip()

    predictions.append({
        "id": qa["id"],
        "prediction_text": pred_answer
    })

    references.append({
        "id": qa["id"],
        "answers": {
            "text": [true_answer],
            "answer_start": [answer_start]
        }
    })

# Compute metrics
results = squad_metric.compute(predictions=predictions, references=references)
print("✅ Evaluation Results:")
print("Exact Match (EM):", results["exact_match"])
print("F1 Score:", results["f1"])
