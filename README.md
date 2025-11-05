# FAQ Bot 🤖

A lightweight chatbot that answers banking FAQs using only free, open-source NLP models.

## 🔗 Live Demo
Access the bot here: [FAQ Bot Live](https://huggingface.co/spaces/thecoderhere/jupiter-faq-bot)

## 🔍 Features
- 87.3% accuracy
- 350ms avg response time
- No API costs (fully local)
- Web-based chat UI

## ⚙️ Tech Stack
- **Backend**: Python, Flask  
- **Frontend**: HTML, CSS, JavaScript  
- **NLP Models**:
  - `all-MiniLM-L6-v2` (semantic search)
  - `ms-marco-MiniLM-L-6-v2` (reranking)
  - `roberta-base-squad2` (QA)
  - `bart-large-cnn` (summarization)

## 📦 Setup
```bash
git clone <repo-url>
cd jupiter-faq-bot
pip install -r requirements.txt
python run.py
