import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever

st.set_page_config(
  page_title="Rag based bot for GPMC using Llamaindex",
  page_icon="🦙",
  layout="centered",
  initial_sidebar_state="auto",
  menu_items=None
)

openai.api_key = st.secrets.openai_key
st.title("Rag Based Bot for The Ahmedabad Municipal Corporation")

if "messages" not in st.session_state.keys():
  st.session_state.messages = [
      {
          "role": "assistant",
          "content": "Welcome! I can help you understand the GPMC Act and AMC procedures in detail. Please ask your question!"
      }
  ]

@st.cache_resource(show_spinner=False)
def load_data():
  reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
  docs = reader.load_data()
  
  # Enhanced system prompt for more detailed responses
  system_prompt = """You are an authoritative expert on the GPMC Act and the Ahmedabad Municipal Corporation. 
  Your responses should be:
  1. Comprehensive and detailed
  2. Include step-by-step procedures when applicable
  3. Quote relevant sections directly from the GPMC Act
  4. Provide specific references (section numbers, chapters)
  5. Break down complex processes into numbered steps
  6. Include any relevant timelines or deadlines
  7. Mention any prerequisites or requirements
  8. Highlight important caveats or exceptions
  
  Always structure your responses in a clear, organized manner using:
  - Bullet points for lists
  - Numbered steps for procedures
  - Bold text for important points
  - Separate sections with clear headings
  
  If multiple interpretations are possible, explain each one clearly."""

  Settings.llm = OpenAI(
      model="gpt-4",  # Using GPT-4 for more detailed and accurate responses
      temperature=0.1,  # Lower temperature for more focused responses
      system_prompt=system_prompt,
  )
  
  index = VectorStoreIndex.from_documents(docs)
  return index

index = load_data()

if "chat_engine" not in st.session_state.keys():
  # Configure retriever for more comprehensive context
  retriever = VectorIndexRetriever(
      index=index,
      similarity_top_k=5,  # Increase the number of relevant chunks retrieved
  )
  
  # Configure response synthesizer for detailed responses
  response_synthesizer = get_response_synthesizer(
      response_mode="tree_summarize",
      verbose=True,
  )
  
  st.session_state.chat_engine = index.as_chat_engine(
      chat_mode="condense_question",
      verbose=True,
      streaming=True,
      retriever=retriever,
      response_synthesizer=response_synthesizer,
  )

# Add a helper function to format the response
def format_response(response):
  # Add markdown formatting for better readability
  formatted_response = response.replace("Step ", "\n### Step ")
  formatted_response = formatted_response.replace("Note:", "\n> **Note:**")
  formatted_response = formatted_response.replace("Important:", "\n> **Important:**")
  return formatted_response

if prompt := st.chat_input("Ask a question about GPMC Act or AMC procedures"):
  st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
      st.markdown(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant"):
      response_stream = st.session_state.chat_engine.stream_chat(prompt)
      response_text = ""
      for response in response_stream.response_gen:
          response_text += response
          st.markdown(format_response(response_text))
      
      message = {
          "role": "assistant",
          "content": format_response(response_text)
      }
      st.session_state.messages.append(message)
