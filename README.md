# User Question Answering System

This project builds a BERT-based QA system that extracts answers from a given passage when a user asks a question.

## Features
- Input: Paragraph + Question
- Output: Extracted answer
- Expecteded Answer(F1 score & EM)
- Model: Pretrained BERT (fine-tuned on SQuAD)
- Deployment: Streamlit app
- View Histroy

## Run the App
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit:
```bash
streamlit run app.py
```

## Example
**Paragraph**: BERT is developed by Google for NLP tasks.  
**Question**: Who developed BERT?  
**Answer**: Google

## Future Improvements
- Document uploads
- Answer highlighting
- Backend API
