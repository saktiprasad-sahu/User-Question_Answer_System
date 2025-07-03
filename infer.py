from transformers import BertTokenizer, BertForQuestionAnswering
import torch

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def get_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    )
    return answer.strip()
