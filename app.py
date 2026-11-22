import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import Counter
import streamlit as st
import torch
from docx import Document
from pypdf import PdfReader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# --- Page Config ---
st.set_page_config(
    page_title="AI Document Intelligence: Sentiment & Summary",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        color: #212529;
        border: 1px solid #e9ecef;
        padding: 18px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .metric-card h4 {
        color: #212529;
        margin-bottom: 10px;
    }
    .badge-pos {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Model Loading with Caching ---
@st.cache_resource(show_spinner="Loading trained models into memory...")
def load_nlp_pipelines():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load sentiment model directly
    sem_tokenizer = AutoTokenizer.from_pretrained("./saved_semantic_model")
    sem_model = AutoModelForSequenceClassification.from_pretrained("./saved_semantic_model").to(device_str)
    sem_model.eval()

    # Load T5 summarizer directly
    sum_tokenizer = AutoTokenizer.from_pretrained("./saved_summarizer_model")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("./saved_summarizer_model").to(device_str)
    sum_model.eval()

    return sem_tokenizer, sem_model, sum_tokenizer, sum_model, device_str


try:
    sem_tokenizer, sem_model, sum_tokenizer, sum_model, device_str = load_nlp_pipelines()
except Exception as e:
    st.error(f"❌ Could not load models.\n\nError: {e}")
    st.stop()


# --- Text Parsing Functions ---
def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    elif filename.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    elif filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    return ""


def chunk_text(text: str, max_words: int = 350) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


# --- Sidebar Parameters ---
st.sidebar.title("⚙️ Inference Controls")
summary_length_mode = st.sidebar.selectbox(
    "Summary Depth",
    ["Standard", "Concise", "Detailed"],
    index=0,
)

max_summary_len = {"Concise": 60, "Standard": 120, "Detailed": 180}[summary_length_mode]
min_summary_len = {"Concise": 20, "Standard": 35, "Detailed": 60}[summary_length_mode]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Execution Device:** `{'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}`"
)


# --- Main Dashboard Header ---
st.title("📑 Semantic Sentiment & Summarization Engine")
st.write(
    "Extract key insights, detect semantic polarity, and generate abstractive summaries from raw text or uploaded documents."
)

input_method = st.radio(
    "Choose Input Method:",
    ("Direct Text Input", "Upload File (.pdf, .docx, .txt)"),
    horizontal=True,
)

raw_text = ""

if input_method == "Direct Text Input":
    raw_text = st.text_area(
        "Enter or paste text block here:",
        height=220,
        placeholder="Type or paste paragraphs here...",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="Upload PDF reports, Word documents, or plain text files.",
    )
    if uploaded_file is not None:
        raw_text = extract_text_from_file(uploaded_file)
        if raw_text:
            with st.expander("📄 Document Raw Preview (First 500 characters)"):
                st.write(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))
        else:
            st.warning("The uploaded file contains no readable text or is an image-only scan.")

# --- Process & Analyze ---
if st.button("🚀 Analyze & Summarize", type="primary"):
    if not raw_text.strip():
        st.warning("Please provide input text or upload a valid readable file.")
    else:
        chunks = chunk_text(raw_text, max_words=350)
        total_words = len(raw_text.split())

        progress_bar = st.progress(0, text="Running semantic analysis...")

        # 1. Semantic Analysis Execution
        sentiment_results = []
        for chunk in chunks:
            sem_inputs = sem_tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device_str)

            with torch.no_grad():
                logits = sem_model(**sem_inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                pred_id = torch.argmax(probs).item()

            label = sem_model.config.id2label.get(
                pred_id, sem_model.config.id2label.get(str(pred_id), f"LABEL_{pred_id}")
            )
            sentiment_results.append({"label": label, "score": probs[pred_id].item()})

        labels = [r["label"] for r in sentiment_results]
        label_counts = Counter(labels)
        dominant_label = label_counts.most_common(1)[0][0]
        avg_score = sum(r["score"] for r in sentiment_results) / len(sentiment_results)

        progress_bar.progress(50, text="Generating document summaries...")

        # 2. Summarization Execution
        summaries = []
        for chunk in chunks:
            chunk_word_count = len(chunk.split())
            if chunk_word_count < 25:
                summaries.append(chunk)
                continue

            target_max = min(max_summary_len, max(min_summary_len + 10, chunk_word_count // 2))
            target_min = min(min_summary_len, target_max - 5)

            inputs = sum_tokenizer(
                "summarize: " + chunk,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(device_str)

            with torch.no_grad():
                summary_ids = sum_model.generate(
                    **inputs,
                    max_new_tokens=target_max,
                    min_new_tokens=target_min,
                    do_sample=False,
                )

            summary_text = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)

        final_summary = " ".join(summaries)
        progress_bar.progress(100, text="Completed!")
        progress_bar.empty()

        st.markdown("---")

        # --- Display Results ---
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.subheader("📊 Semantic Insights")

            badge_class = (
                "badge-pos"
                if dominant_label.upper() in ["POSITIVE", "LABEL_1"]
                else "badge-neg"
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>Overall Sentiment</h4>
                    <span class="{badge_class}">{dominant_label.upper()}</span>
                    <p style="margin-top: 10px; color: #555;">Mean Confidence: <strong>{avg_score * 100:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.metric("Total Document Word Count", f"{total_words:,}")
            st.metric("Chunks Processed", len(chunks))

            if len(chunks) > 1:
                st.write("**Chunk Distribution:**")
                for lbl, count in label_counts.items():
                    st.write(f"- `{lbl}`: {count} chunk(s) ({(count / len(chunks)) * 100:.0f}%)")

        with col2:
            st.subheader("📝 Abstractive Summary")
            st.write(final_summary)

            summary_word_count = len(final_summary.split())
            reduction_ratio = (
                ((total_words - summary_word_count) / total_words * 100)
                if total_words > 0
                else 0
            )
            st.caption(
                f"Summary length: **{summary_word_count} words** (Reduced document length by **{reduction_ratio:.1f}%**)"
            )

            # Export Button
            report_text = (
                f"DOCUMENT INTELLIGENCE REPORT\n{'=' * 30}\n\n"
                f"Overall Polarity: {dominant_label} ({avg_score * 100:.2f}%)\n"
                f"Total Original Words: {total_words}\n\n"
                f"SUMMARY:\n{final_summary}"
            )
            st.download_button(
                label="📥 Download Summary Report (.txt)",
                data=report_text,
                file_name="analysis_summary.txt",
                mime="text/plain",
            )