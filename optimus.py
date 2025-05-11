import os
import spacy
import streamlit as st
from transformers import pipeline

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load Hugging Face zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Classification labels
LABELS = ["malware", "benign", "suspicious", "backdoor", "vulnerability"]

# Function to analyze text using spaCy and Hugging Face
def analyze_text(content, filename):
    findings = []
    try:
        doc = nlp(content)
        for sent in doc.sents:
            sentence = sent.text.strip()
            result = classifier(sentence, LABELS)
            top_label = result['labels'][0]
            top_score = result['scores'][0]

            if top_label in ["malware", "suspicious", "backdoor"] and top_score > 0.7:
                findings.append({
                    "file": filename,
                    "label": top_label,
                    "score": round(top_score, 3),
                    "sentence": sentence
                })
    except Exception as e:
        st.error(f"Error analyzing {filename}: {str(e)}")
    return findings

# Streamlit UI
st.set_page_config(page_title="AI-Powered Firmware Malware Scanner", layout="centered")
st.title("🛡️ Firmware Malware Scanner with NLP + Transformers")

st.markdown("Upload firmware or config files. This app uses AI to detect potentially **malicious or vulnerable** lines in your firmware.")

uploaded_files = st.file_uploader("Upload files (text/scripts/configs)", type=None, accept_multiple_files=True)

if uploaded_files:
    st.subheader("📊 Analysis Results")
    all_findings = []

    for file in uploaded_files:
        filename = file.name
        try:
            content = file.read().decode(errors="ignore")
            findings = analyze_text(content, filename)
            all_findings.extend(findings)
        except:
            st.warning(f"⚠️ Skipped binary or unreadable file: {filename}")

    if all_findings:
        st.success(f"⚠️ Potential threats found in {len(set(f['file'] for f in all_findings))} file(s).")
        for item in all_findings:
            st.markdown(f"""
                **File:** `{item['file']}`  
                **Prediction:** `{item['label']}`  
                **Confidence:** `{item['score']}`  
                **Sentence:**  
                > _{item['sentence']}_  
                ---
            """)
    else:
        st.success("✅ No suspicious content detected in uploaded files.")
