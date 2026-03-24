from collections import defaultdict

USER_MEMORY = defaultdict(list)

def add_to_memory(user_id, query, response):
    USER_MEMORY[user_id].append((query, response))
    USER_MEMORY[user_id] = USER_MEMORY[user_id][-3:]

def get_memory(user_id):
    history = USER_MEMORY.get(user_id, [])
    formatted = ""
    for q, r in history:
        formatted += f"User: {q}\nBot: {r}\n"
    return formatted