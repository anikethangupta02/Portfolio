import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-120b"
DATA_PATH = "data/"
CHROMA_PATH = "db/"