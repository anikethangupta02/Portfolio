# 🤖 GenAI Telegram + Streamlit Bot (RAG + Vision)

A lightweight **multi-modal GenAI bot** that supports:

* 💬 **Text-based Q&A (RAG)** using local documents
* 🖼 **Image description** using a vision model
* 🤖 Works via **Telegram bot + Streamlit UI**

---

# 🚀 Features

* ✅ Retrieval-Augmented Generation (RAG)
* ✅ Image captioning + keyword tagging
* ✅ Telegram bot interface
* ✅ Streamlit ChatGPT-style UI
* ✅ Memory (last 3 user interactions)
* ✅ Source attribution for answers
* ✅ Lightweight & runs locally

---

# 🧠 Tech Stack

| Component     | Technology                       |
| ------------- | -------------------------------- |
| LLM           | Groq API (`openai/gpt-oss-120b`) |
| RAG Framework | LangChain                        |
| Embeddings    | all-MiniLM-L6-v2                 |
| Vector DB     | Chroma                           |
| Vision Model  | BLIP (Salesforce)                |
| Bot           | python-telegram-bot              |
| UI            | Streamlit                        |

---


# ⚙️ Setup Instructions

## 1. Clone Repo

```bash
git clone <your-repo-url>
cd genai-bot
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Add API Keys

Edit `.env`:

```python
TELEGRAM_TOKEN = "your_telegram_token"
GROQ_API_KEY = "your_groq_api_key"
```

---

## 4. Add Data

Place `.md` or `.txt` files inside:

```
data/
```

---

## 5. Reset Vector DB (first run only)

```bash
rm -rf db/
```

---

# ▶️ Running the Application

## Run Telegram Bot

```bash
python app.py
```

Then open your bot in Telegram:

```
/ask What are work hours?
(send image for captioning)
```

---

## Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 🧪 Example Usage

### Text Query

```
/ask What is leave policy?
```

👉 Output:

```
💡 Leave must be applied 2 days in advance.
📚 Sources: faq.md
```

---

### Image Upload

👉 Upload an image

```
🖼 A young boy running on a grassy field outdoors
🏷 boy, running, grass, outdoors
```

---

# 🧠 How It Works

## RAG Pipeline

1. Documents are loaded from `/data`
2. Split into chunks
3. Embedded using sentence-transformers
4. Stored in Chroma vector DB
5. Query → retrieve top-k chunks
6. Context sent to Groq LLM → answer generated

---

## Vision Pipeline

1. User uploads image
2. BLIP model generates caption
3. Keywords extracted from caption
4. Response returned to user

---
