import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Title and Introduction
st.title("AI Chatbot for Your Project")
st.markdown("This is a simple chatbot interface for answering questions related to your project.")

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    # Replace 'gpt2' with your preferred model or any fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Function to Generate Response
def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Chat Interface
if "history" not in st.session_state:
    st.session_state.history = []

# User Input
with st.form("chat_form"):
    user_input = st.text_input("Ask a question about your project:")
    submit_button = st.form_submit_button("Send")

# Display Chat History
for message in st.session_state.history:
    st.write(f"**{message['role']}**: {message['content']}")

# Process Input
if submit_button and user_input:
    st.session_state.history.append({"role": "User", "content": user_input})
    bot_response = generate_response(user_input)
    st.session_state.history.append({"role": "Bot", "content": bot_response})
    st.write(f"**Bot**: {bot_response}")
