# XEV 9E Car RAG Agent

This project is a Retrieval-Augmented Generation (RAG) agent for answering questions about your car manual using OpenAI's GPT models. It provides both an API (FastAPI) and a user-friendly web UI (Streamlit).

## Features
- Ask questions about your car manual PDF
- Uses OpenAI GPT-3.5/4 for answers
- FastAPI backend for API access
- Streamlit frontend for chat UI

## Prerequisites
- Python 3.8+
- OpenAI API key
- The car manual PDF (`car_manual.pdf`) in the project root

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

## Running the FastAPI Backend
Start the API server:
```bash
uvicorn fastapi_app:app --reload
```
The API will be available at `http://localhost:8000`.

## Running the Streamlit UI
In a new terminal, run:
```bash
streamlit run streamlit_app.py
```
Open the displayed URL (usually http://localhost:8501) in your browser.

## Usage
1. Enter your OpenAI API key in the Streamlit UI.
2. Ask questions about your car manual.
3. View answers and chat history.

## CLI Usage
You can also run the agent in the terminal:
```bash
python car_agent.py
```

## File Overview
- `car_agent.py`: CLI chat with the RAG agent
- `rag_agent.py`: Core RAG agent logic (reusable)
- `fastapi_app.py`: FastAPI backend
- `streamlit_app.py`: Streamlit web UI
- `car_manual.pdf`: Your car manual (PDF)
- `requirements.txt`: Python dependencies

## Notes
- Make sure your OpenAI API key has sufficient quota.
- The first question may take longer as the PDF is processed and indexed.




