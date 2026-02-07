class AgentMemory:
    def __init__(self):
        self.logs=[]

    def add(self, entry):
        self.logs.append(entry)

    def get_all(self):
        return self.logs