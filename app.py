import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖",
    layout="centered"
)

# Mandatory title: Name and Student ID
st.title("Dawar Shariq | Student ID: B04-0923-000022")

st.header("AI Text Generator")
st.write(
    "Enter a prompt and let the AI generate a continuation "
    "using a pretrained Hugging Face model."
)

# Load and cache the model so it is not reloaded on every interaction
MODEL_NAME = "distilgpt2"

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model=MODEL_NAME
    )

# Load the model
with st.spinner("Loading AI model..."):
    generator = load_model()

st.success(f"Model loaded: {MODEL_NAME}")

# Input widget
prompt = st.text_area(
    "Enter your starting text:",
    placeholder="Artificial intelligence can help people...",
    height=120
)

# Required feature: control maximum generated tokens
max_new_tokens = st.slider(
    "Maximum new tokens",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

# Button and model call
if st.button("Generate Text", type="primary"):
    if prompt.strip():
        with st.spinner("Generating text..."):
            result = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.8,
                pad_token_id=generator.tokenizer.eos_token_id
            )

        # Display only the generated continuation
        full_text = result[0]["generated_text"]
        continuation = full_text[len(prompt):]

        st.subheader("Generated Continuation")
        if continuation.strip():
            st.write(continuation)
        else:
            st.write(full_text)

    else:
        st.warning("Please enter a prompt before generating text.")

# Model information
with st.expander("About this application"):
    st.write("Model: distilgpt2")
    st.write("Pipeline task: text-generation")
    st.write("Framework: Hugging Face Transformers")
    st.write("Interface: Streamlit")
