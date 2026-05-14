import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "rinna/japanese-gpt2-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, prompt: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

st.title("日本語チャットボット")

if "messages" not in st.session_state:
    st.session_state.messages = []

tokenizer, model = load_model()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("メッセージを入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("生成中..."):
            response = generate(tokenizer, model, prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
