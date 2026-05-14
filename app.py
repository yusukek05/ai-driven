import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODELS = {
    "gpt2 small (124M)": "openai-community/gpt2",
    "gpt2 medium (355M)": "openai-community/gpt2-medium",
    "gpt2 large (774M)": "openai-community/gpt2-large",
}

@st.cache_resource
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
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

st.title("GPT-2 Chatbot")

with st.sidebar:
    st.header("モデル設定")
    model_label = st.selectbox("モデルを選択", list(MODELS.keys()))
    model_name = MODELS[model_label]
    if "large" in model_label:
        st.warning("This model may run out of memory on Streamlit Cloud.")

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
        with st.spinner("生成中..."):
            response = generate(tokenizer, model, prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
