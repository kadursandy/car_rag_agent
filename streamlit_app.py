import streamlit as st
import requests

st.title("Car Manual RAG Agent Chat")

# Store API key and chat history in session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.session_state.api_key = st.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.api_key)

question = st.text_input("Ask a question about your car manual:")

if st.button("Ask") and question and st.session_state.api_key:
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question, "api_key": st.session_state.api_key}
    )
    if response.status_code == 200:
        answer = response.json()["answer"]
        st.session_state.chat_history.append((question, answer))
    else:
        st.error(f"Error: {response.text}")

# Display chat history
if st.session_state.chat_history:
    st.write("## Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Agent:** {a}") 