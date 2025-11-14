import streamlit as st
import pandas as pd
import speech_recognition as sr
from transformers import pipeline

# -----------------------------
# 1️⃣ Load Dataset + Build Context
# -----------------------------
@st.cache_data
def load_context():
    df = pd.read_csv("top_20_movie_dataset.csv").fillna("")
    context = df.astype(str).agg('. '.join, axis=1)
    context_text = "\n".join(context)
    return context_text

context_text = load_context()

# -----------------------------
# 2️⃣ Load FLAN-T5 Model (Local)
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

# -----------------------------
# 3️⃣ Ask Function
# -----------------------------
def ask(query):
    prompt = (
        f"Use the following data to answer the question. "
        f"If not available, say 'No information found.'\n\n"
        f"Data:\n{context_text}\n\n"
        f"Question: {query}\nAnswer:"
    )

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )[0]['generated_text']

    return result.strip()

# -----------------------------
# 4️⃣ Streamlit UI
# -----------------------------
st.title("Movie_Recommendation_Chatbot with Voice Input")

st.write("Click the button below and start speaking.")

if st.button("🎧 Start Recording"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = r.listen(source)
        st.success("Recording complete!")

    st.write("Transcribing...")

    try:
        text = r.recognize_google(audio)
        st.success(" You said:")
        st.write(text)

        # Call model
        st.write(" Generating answer...")
        answer = ask(text)

        st.subheader(" AI Answer")
        st.write(answer)

    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError:
        st.error("Google Speech API error.")