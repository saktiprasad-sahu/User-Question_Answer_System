from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

dataset = load_dataset("json", data_files={"train": "data/squad_sample.json"}, field="data")

def preprocess(example):
    context = example["paragraphs"][0]["context"]
    question = example["paragraphs"][0]["qas"][0]["question"]
    answers = example["paragraphs"][0]["qas"][0]["answers"][0]
    return {
        "context": context,
        "question": question,
        "answers": {
            "text": [answers["text"]],
            "answer_start": [answers["answer_start"]],
        }
    }

dataset = dataset["train"].map(preprocess)
tokenized = dataset.map(lambda x: tokenizer(x["question"], x["context"], truncation=True, padding="max_length"), batched=True)

args = TrainingArguments(
    output_dir="./models/my-bert-qa-model",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
model.save_pretrained("models/my-bert-qa-model")
tokenizer.save_pretrained("models/my-bert-qa-model")
