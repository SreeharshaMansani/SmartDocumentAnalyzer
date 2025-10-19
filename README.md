# 🚀 Smart Document Analyzer

## 📄 Overview
Smart Document Analyzer is a **powerful Python tool** for dynamically analyzing and summarizing documents including **PDF**, **DOCX**, and plain text. It identifies key topics, generates rich summaries, extracts clean named entities, performs sentiment analysis, and delivers an insightful interpretation of the document’s core subject.

---

## ✨ Features
- 🗂️ **Multi-format Input Support:** PDF, DOCX, and plain text
- 🔍 **Dynamic Document Overview:** Zero-shot topic classification
- 📝 **Advanced Summarization:** Transformer-based detailed summaries (BART)
- 🧑‍🤝‍🧑 **Named Entity Recognition:** spaCy transformer with deduplication & pronoun filtering
- 😊 **Sentiment Analysis:** Polarity with confidence scores
- 📚 **Topic Modeling:** Zero-shot classification with broad categories
- 🔑 **Keyword Extraction:** TF-IDF based
- 🧠 **Core Subject Interpretation:** Combined insights from sentiment, topics, keywords
- 💻 **User-Friendly Console Output & Robust Error Handling**

---

## ⚙️ Installation

Run the following commands:
pip install transformers torch sentencepiece spacy spacy-transformers pdfplumber python-docx scikit-learn
python -m spacy download en_core_web_trf


---

## 🚀 Usage

1. Run the Python script.
2. Choose to upload a **PDF or DOCX** file, or **paste text manually**.
3. Review the output containing:
   - 📋 **Document Overview**
   - 🙂 **Sentiment Analysis**
   - 🔍 **Topic Classification**
   - 🔑 **Top Keywords**
   - 🏷️ **Named Entities**
   - 🧠 **Core Subject Interpretation**

---

## 🔧 Customization

- Modify token limits (`MAX_SUMMARY_TOKENS`, etc.) to fit document sizes.
- Update topic labels to better match domain-specific needs.
- Customize entity filtering and interpretation strings for tailored output.

---

## ⚠️ Limitations

- Long documents are truncated due to model token limits; chunking recommended for full analysis.
- SpaCy transformer NER needs substantial resources; performance varies by system.
- Zero-shot topic detection depends on predefined labels; add labels for niche domains.


