import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
import re

# Set up the Streamlit page configuration
st.set_page_config(
  page_title="Rag Based Bot for GPMC using Llamaindex",
  page_icon="🦙",
  layout="centered",
  initial_sidebar_state="auto"
)

# Initialize OpenAI API key
openai.api_key = st.secrets.get("openai_key", None)
if not openai.api_key:
  st.error("OpenAI API key is missing. Please add it to the Streamlit secrets.")
  st.stop()

st.title("Rag Based Bot for The Ahmedabad Municipal Corporation")

# Initialize session state for messages
if "messages" not in st.session_state:
  st.session_state.messages = [
      {
          "role": "assistant",
          "content": "Welcome! I can help you understand the GPMC Act and AMC procedures in detail. Please ask your question!"
      }
  ]

@st.cache_resource(show_spinner=False)
def load_data():
  try:
      node_parser = SimpleNodeParser.from_defaults(
          chunk_size=512,
          chunk_overlap=50,
      )
      
      reader = SimpleDirectoryReader(
          input_dir="./data",
          recursive=True,
          filename_as_id=True
      )
      docs = reader.load_data()
      
      system_prompt = """You are an authoritative expert on the GPMC Act and the Ahmedabad Municipal Corporation. 
      Your responses should be:
      1. Comprehensive and detailed
      2. Include step-by-step procedures when applicable
      3. Quote relevant sections directly from the GPMC Act
      4. Provide specific references (section numbers, chapters, and page numbers)
      5. Break down complex processes into numbered steps
      6. Include any relevant timelines or deadlines
      7. Mention any prerequisites or requirements
      8. Highlight important caveats or exceptions
      
      For every fact or statement, include a reference to the source document and page number in this format:
      [Source: Document_Name, Page X]
      
      Always structure your responses in a clear, organized manner using:
      - Bullet points for lists
      - Numbered steps for procedures
      - Bold text for important points
      - Separate sections with clear headings"""

      Settings.llm = OpenAI(
          model="gpt-4",
          temperature=0.1,
          system_prompt=system_prompt,
      )
      
      index = VectorStoreIndex.from_documents(
          docs,
          node_parser=node_parser,
          show_progress=True
      )
      return index
  except Exception as e:
      st.error(f"Error loading data: {e}")
      st.stop()

def extract_references(text):
  pattern = r'$Source: ([^,]+), Page (\d+)$'
  matches = re.finditer(pattern, text)
  
  for match in matches:
      doc_name, page = match.groups()
      link = f'<a href="data/{doc_name}.pdf#page={page}" target="_blank">[Source: {doc_name}, Page {page}]</a>'
      text = text.replace(match.group(0), link)
  
  return text

def format_response(response):
  formatted_response = response.replace("Step ", "\n### Step ")
  formatted_response = formatted_response.replace("Note:", "\n> **Note:**")
  formatted_response = formatted_response.replace("Important:", "\n> **Important:**")
  formatted_response = extract_references(formatted_response)
  return formatted_response

# Load the index
index = load_data()

# Initialize chat engine
if "chat_engine" not in st.session_state:
  st.session_state.chat_engine = index.as_chat_engine(
      chat_mode="condense_question",
      verbose=True
  )

# Sidebar for reference documents
with st.sidebar:
  st.header("📚 Reference Documents")
  st.write("Click on references in the chat to view source documents.")

# Chat interface
if prompt := st.chat_input("Ask a question about GPMC Act or AMC procedures"):
  st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
      st.markdown(message["content"], unsafe_allow_html=True)

# Generate new response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant"):
      try:
          # Get the complete response
          response = st.session_state.chat_engine.chat(prompt)
          formatted_response = format_response(response.response)
          
          # Display the complete response
          st.markdown(formatted_response, unsafe_allow_html=True)
          
          # Append the response to the message history
          message = {
              "role": "assistant",
              "content": formatted_response
          }
          st.session_state.messages.append(message)
      except Exception as e:
          st.error(f"Error generating response: {e}")

# Add CSS for better formatting
st.markdown("""
<style>
a {
    color: #0078ff;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)
