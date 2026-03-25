import streamlit as st
from processor import get_chatbot_response

# Page Layout
st.set_page_config(page_title="AI Tourism Assistant", page_icon="🌍")
st.title("🌍 Tourism AI Chatbot")
st.markdown("Find your next travel destination using Natural Language Processing.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ex: I want a nature trip in Egypt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyzing destinations..."):
                response = get_chatbot_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")