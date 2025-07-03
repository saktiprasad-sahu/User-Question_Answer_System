import csv
import os
from datetime import datetime

HISTORY_FILE = "qa_history.csv"

def save_to_history(context, question, predicted, expected, f1, em):
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "context", "question", "predicted_answer", "expected_answer", "F1", "EM"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            context,
            question,
            predicted,
            expected,
            f"{f1:.2f}" if f1 is not None else "",
            f"{em:.2f}" if em is not None else ""
        ])

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, newline='', encoding="utf-8") as f:
            return list(csv.reader(f))[1:]  # Skip header
    return []
