from langchain_groq import ChatGroq

class WriterAgent:

    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b")
        
    def write(self, research_data: str):
        prompt= f"""Explaination based on research: {research_data}"""
        return self.llm.invoke(prompt).content.strip()