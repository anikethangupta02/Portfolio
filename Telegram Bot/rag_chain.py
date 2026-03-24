import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from groq import Groq
from config import DATA_PATH, CHROMA_PATH, MODEL_NAME
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def load_docs():
    docs = []
    for file in os.listdir(DATA_PATH):
        with open(os.path.join(DATA_PATH, file), "r") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": file}))
    return docs


splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = Chroma.from_documents(
    documents=splitter.split_documents(load_docs()),
    embedding=embedding,
    persist_directory=CHROMA_PATH
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use the context to answer the question.

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer clearly and concisely.
""")


def call_llm(prompt_text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.choices[0].message.content


def generate_answer(query, history=""):
    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata["source"] for d in docs]))

    final_prompt = prompt.format(
        context=context,
        history=history,
        question=query
    )

    answer = call_llm(final_prompt)

    return answer, sources