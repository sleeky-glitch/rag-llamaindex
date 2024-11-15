import streamlit as st
    )
    
    # Create chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        streaming=True,
        query_engine=query_engine
    )

# Sidebar for document navigation
with st.sidebar:
    st.header("📚 Reference Documents")
    st.write("Click on references in the chat to view source documents.")

# Chat interface
if prompt := st.chat_input("Ask a question about GPMC Act or AMC procedures"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Generate new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        response_text = ""
        for response in response_stream.response_gen:
            response_text += response
            st.markdown(format_response(response_text), unsafe_allow_html=True)
        
        message = {
            "role": "assistant",
            "content": format_response(response_text)
        }
        st.session_state.messages.append(message)

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

