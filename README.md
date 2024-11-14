# RAG-Based Chatbot for Ahmedabad Municipal Corporation

A Retrieval Augmented Generation (RAG) based chatbot built using LlamaIndex and Streamlit to answer questions about the GPMC Act of Ahmedabad Municipal Corporation (AMC).

## Description

This project implements an AI-powered chatbot that can answer questions related to the GPMC Act and AMC policies. It uses:
- LlamaIndex for document indexing and retrieval
- OpenAI's GPT-3.5-turbo model for generating responses
- Streamlit for the web interface

The chatbot provides detailed, fact-based answers with references from source documents wherever possible.

## Features

- Interactive chat interface
- Context-aware responses
- Document-grounded answers
- Procedural step explanations where applicable
- Source references and quotes from documents
- Streaming responses for better user experience

## Dependencies

The project requires the following Python packages:
- streamlit
- openai
- llama-index
- nltk

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/sleeky-glitch/rag-llamaindex.git
cd rag-llamaindex
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `secrets.toml` file in the `.streamlit` directory with your OpenAI API key:
```toml
openai_key = "your-api-key-here"
```

4. Place your source documents in the `data` directory

5. Run the Streamlit app:
```bash
streamlit run streamlit.py
```

## Usage

1. Once the application is running, you'll see a chat interface with a welcome message
2. Type your question about AMC or GPMC Act in the chat input
3. The chatbot will process your question and provide a detailed response
4. Continue the conversation by asking follow-up questions

## Project Structure

```
.
├── data/               # Directory for source documents
├── .gitignore         # Git ignore file
├── requirements.txt   # Project dependencies
├── streamlit.py      # Main application file
└── README.md         # Project documentation
```

## Technical Details

- The chatbot uses LlamaIndex's VectorStoreIndex for efficient document retrieval
- Responses are generated using GPT-3.5-turbo with a temperature of 0.2 for consistent outputs
- The system uses a custom prompt to ensure responses are focused on AMC and GPMC Act
- Chat history is maintained during the session for context-aware responses

## Notes

- Make sure to keep your OpenAI API key secure and never commit it to version control
- The quality of responses depends on the documents provided in the data directory
- The system is designed to provide factual information based on source documents only