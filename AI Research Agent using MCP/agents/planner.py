from langchain_groq import ChatGroq

class PlannerAgent:

    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b")
    
    def plan(self,task: str):
        prompt= f"""Break this task into clear steps: Task:{task}"""
        return self.llm.invoke(prompt).content.split("\n")