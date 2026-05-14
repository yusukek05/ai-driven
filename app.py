import threading
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch

MODELS = {
    "gpt2 small (124M)": "openai-community/gpt2",
    "gpt2 medium (355M)": "openai-community/gpt2-medium",
}

@st.cache_resource
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    return tokenizer, model

def stream_generate(tokenizer, model, prompt: str, max_new_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    return streamer

st.title("GPT-2 Chatbot")

with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Select model", list(MODELS.keys()))
    model_name = MODELS[model_label]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_model" not in st.session_state:
    st.session_state.current_model = model_name

if st.session_state.current_model != model_name:
    st.session_state.messages = []
    st.session_state.current_model = model_name

tokenizer, model = load_model(model_name)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        streamer = stream_generate(tokenizer, model, prompt)
        response = st.write_stream(streamer)

    st.session_state.messages.append({"role": "assistant", "content": response})
