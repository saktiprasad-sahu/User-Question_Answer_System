import streamlit as st
from infer import get_answer
from evaluate import load
from history import save_to_history, load_history

# Initialize metric
metric = load("squad")

st.set_page_config(page_title="User QA System", layout="centered")
st.title("📘 User Question Answering System")

st.markdown("**Enter a paragraph and ask a question. The system will extract the answer using BERT.**")

context = st.text_area("Enter the paragraph (context):", height=200)
question = st.text_input("Ask a question from the paragraph:")
expected_answer = st.text_input("✅ (Optional) Enter expected answer (for F1/EM score):")

if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Please fill in both context and question.")
    else:
        # Generate prediction
        answer = get_answer(question, context)
        st.success(f"**Answer:** {answer}")

        f1 = em = None
        # If expected answer provided → calculate scores
        if expected_answer.strip() != "":
            predictions = [{"id": "1", "prediction_text": answer}]
            references = [{"id": "1", "answers": {"text": [expected_answer], "answer_start": [0]}}]
            results = metric.compute(predictions=predictions, references=references)
            f1 = results["f1"]
            em = results["exact_match"]

            st.markdown(f"📊 **F1 Score:** `{f1:.2f}`")
            st.markdown(f"✅ **Exact Match (EM):** `{em:.2f}`")

        # Save to history
        save_to_history(context, question, answer, expected_answer, f1, em)

# Add a history viewer
with st.expander("📜 View Search History"):
    history_data = load_history()
    if history_data:
        import pandas as pd
        df = pd.DataFrame(history_data, columns=["Timestamp", "Context", "Question", "Predicted Answer", "Expected Answer", "F1", "EM"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history yet. Try asking a question first.")
