'''installations
!pip install transformers torch sentencepiece
!pip install spacy
!pip install spacy-transformers
!python -m spacy download en_core_web_trf
!pip install pdfplumber
!pip install python-docx
!pip install scikit-learn'''




from transformers import pipeline
import pdfplumber
import docx
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Configurations
MAX_SUMMARY_TOKENS = 1024
MAX_NER_TOKENS = 512
MAX_SENTIMENT_TOKENS = 512

# Load spaCy transformer model for NER
print("Loading spaCy model for NER...")
nlp_spacy = spacy.load("en_core_web_trf")
print("spaCy model loaded.")

# Load HuggingFace pipelines
print("Loading HuggingFace NLP models...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("Models loaded successfully.")


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def summarize_text(text):
    if len(text) > MAX_SUMMARY_TOKENS * 5:
        text = text[:MAX_SUMMARY_TOKENS * 5]
    summary = summarizer(text, max_length=300, min_length=100, do_sample=False)
    return summary[0]['summary_text']


def extract_entities(text):
    pronouns = {"i", "me", "he", "she", "they", "we", "you", "him", "her", "us", "them"}
    doc = nlp_spacy(text[:MAX_NER_TOKENS * 5])
    seen = set()
    entities = []
    for ent in doc.ents:
        ent_text = ent.text.strip()
        ent_text_lower = ent_text.lower()
        if (re.match(r"^[A-Za-z][A-Za-z0-9&\-. ]+$", ent_text) and 
            ent_text_lower not in pronouns and 
            ent_text_lower not in seen):
            entities.append({"word": ent_text, "entity_group": ent.label_})
            seen.add(ent_text_lower)
    return entities


def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text[:20000]])
    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]


def analyze_sentiment(text):
    return sentiment_analyzer(text[:MAX_SENTIMENT_TOKENS])[0]


def analyze_topics(text, top_n=3):
    candidate_labels = [
        "Finance", "Technology", "Legal", "Marketing", "Human Resources",
        "Research", "Customer Service", "Product Development",
        "Education", "Healthcare", "Resume", "Email", "Report", "Miscellaneous"
    ]
    results = topic_classifier(text[:2000], candidate_labels)
    topic_scores = sorted(zip(results['labels'], results['scores']), key=lambda x: x[1], reverse=True)
    return topic_scores[:top_n]


def entity_summary_table(entities):
    summary = {}
    for e in entities:
        group = e['entity_group']
        summary[group] = summary.get(group, 0) + 1
    return summary


def print_entities(entities):
    entities_sorted = sorted(entities, key=lambda e: e['entity_group'])
    for entity in entities_sorted:
        print(f" - {entity['word']:30} ({entity['entity_group']})")


def generate_dynamic_doc_description(text):
    candidate_labels = [
        "Finance", "Technology", "Legal", "Marketing", "Human Resources",
        "Research", "Customer Service", "Product Development",
        "Education", "Healthcare", "Resume", "Email", "Report", "Miscellaneous"
    ]
    topic_results = topic_classifier(text[:2000], candidate_labels)
    top_topics = topic_results['labels'][:3]

    summary_text = summarizer(text[:MAX_SUMMARY_TOKENS * 5], max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    topics_str = ", ".join(top_topics)
    description = f"This document primarily covers topics such as {topics_str}. Key details include: {summary_text}"
    return description


def generate_proper_core_subject_interpretation(sentiment, top_topics, keywords):
    label = sentiment['label']
    score = sentiment['score']
    sentiment_map = {
        ('POSITIVE', True): "expresses an optimistic and confident view",
        ('NEGATIVE', True): "conveys concerns or critical viewpoints",
        ('POSITIVE', False): "shows a generally positive tone",
        ('NEGATIVE', False): "reflects some negativity or caution",
    }
    strong_sentiment = score > 0.85
    sentiment_phrase = sentiment_map.get((label, strong_sentiment), "maintains a neutral and factual tone")

    topics_str = ", ".join(top_topics[:3])
    keywords_str = ", ".join(keywords[:4])
    interpretation = (
        f"The document primarily discusses topics related to {topics_str}. "
        f"The core themes are characterized by keywords such as {keywords_str}. "
        f"The sentiment analysis {sentiment_phrase}, indicating how these subjects are treated."
    )
    return interpretation


def main():
    print("\n=== Smart Document Analyzer (Updated Complete) ===\n")
    print("Choose input method:\n1. Upload a file (PDF/DOCX)\n2. Paste text manually")
    choice = input("Enter 1 or 2: ").strip()
    text = ""

    if choice == "1":
        import os
        from google.colab import files
        print("\nUpload your PDF or DOCX file:")
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return
        filename = list(uploaded.keys())[0]
        print(f"Processing: {filename}")
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(filename)
        elif filename.lower().endswith(".docx"):
            text = extract_text_from_docx(filename)
        else:
            print("Unsupported file type! Use PDF or DOCX.")
            return

    elif choice == "2":
        print("\nPaste your text below. Press Enter twice to finish:")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                break
            lines.append(line)
        text = "\n".join(lines)

    else:
        print("Invalid choice!")
        return

    if not text.strip():
        print("No text provided.")
        return

    print("\n=== Document Overview ===")
    try:
        doc_description = generate_dynamic_doc_description(text)
        print(doc_description)
    except Exception as e:
        print(f"Document description generation failed: {e}")

    print("\n=== Sentiment Analysis ===")
    try:
        sentiment_result = analyze_sentiment(text)
        print(f"Overall Sentiment: {sentiment_result['label']} (Confidence: {sentiment_result['score']:.4f})")
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")

    print("\n=== Topic Modeling ===")
    try:
        topics_result = analyze_topics(text)
        for topic, score in topics_result:
            print(f" - {topic}: {score:.2f}")
        top_topics = [topic for topic, _ in topics_result]
    except Exception as e:
        print(f"Topic modeling failed: {e}")
        top_topics = []

    print("\n=== Keywords ===")
    try:
        keywords_result = extract_keywords(text)
        print(", ".join(keywords_result))
    except Exception as e:
        print(f"Keyword extraction failed: {e}")

    print("\n=== Named Entities ===")
    try:
        entities_result = extract_entities(text)
        summary = entity_summary_table(entities_result)
        if summary:
            print("Entity Counts by Type:")
            for group, count in summary.items():
                print(f" - {group}: {count}")
        else:
            print("No entities found.")
        print("\nEntities List:")
        print_entities(entities_result)
    except Exception as e:
        print(f"Entity extraction failed: {e}")

    print("\n=== Core Subject Interpretation ===")
    if top_topics and keywords_result and sentiment_result:
        try:
            interpretation = generate_proper_core_subject_interpretation(
                sentiment_result, top_topics, keywords_result
            )
            print(interpretation)
        except Exception as e:
            print(f"Interpretation generation failed: {e}")
    else:
        print("Insufficient data for core subject interpretation.")

    print("\n=== Analysis Complete ===\n")


if __name__ == "__main__":
    main()
