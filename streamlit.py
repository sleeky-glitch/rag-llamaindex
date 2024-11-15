import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
import re

st.set_page_config(
    page_title="Rag based bot for GPMC using Llamaindex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

openai.api_key = st.secrets.openai_key
st.title("Rag Based Bot for The Ahmedabad Municipal Corporation")

# Initialize session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Welcome! I can help you understand the GPMC Act and AMC procedures in detail. Please ask your question!"
