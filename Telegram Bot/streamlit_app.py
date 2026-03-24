import streamlit as st
from rag_chain import generate_answer
from vision import describe_image
from memory import add_to_memory, get_memory
import tempfile

st.set_page_config(page_title="GenAI Bot", layout="wide")

st.title("GenAI Chat UI")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_id = "web_user"


user_input = st.chat_input("Ask a question or upload image below...")


for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)


if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(("user", user_input))

    with st.spinner("Thinking..."):
        history = get_memory(user_id)
        answer, sources = generate_answer(user_input, history)

        add_to_memory(user_id, user_input, answer)

    full_response = f"{answer}\n\n Sources: {', '.join(sources)}"

    st.chat_message("assistant").write(full_response)
    st.session_state.messages.append(("assistant", full_response))


uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    with st.spinner("Analyzing image..."):
        caption, tags = describe_image(path)

    response = f" {caption}\n {', '.join(tags)}"

    st.chat_message("assistant").write(response)
    st.session_state.messages.append(("assistant", response))