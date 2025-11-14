import streamlit as st
import pandas as pd
import speech_recognition as sr
from transformers import pipeline

# -----------------------------
# 1️⃣ Load CSV + Build Context
# -----------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv("top_20_movie_dataset.csv").fillna("")
    context = df.astype(str).agg('. '.join, axis=1)
    context_text = "\n".join(context)
    return df, context_text

df, context_text = load_dataset()

# -----------------------------
# 2️⃣ Load FLAN-T5 Model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

generator = load_model()

# -----------------------------
# 3️⃣ Ask Function (LLM fallback)
# -----------------------------
def ask(query):
    prompt = f"""
Use the following movie data to answer the question. 
If no matching movies are found, say 'No information found.'

Data:
{context_text}

Question: List movies and their ratings for genre: {query}
Answer:
"""
    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )[0]['generated_text']

    return result.strip()

# -----------------------------
# 4️⃣ Streamlit UI
# -----------------------------
st.title("Movie_Recommendation_Chatbot (Text + Voice)")

st.subheader("Text Based Recommendation")
genre = st.text_input("Enter the genre of the movie you want to watch")

if genre:
    filtered = df[df['Genre'].str.contains(genre, case=False, na=False)]

    if not filtered.empty:
        output = ""
        for _, row in filtered.iterrows():
            output += f"- {row['Title']} (Rating: {row['Rating']})\n"

        st.write("### Recommended Movies:")
        st.write(output)

    else:
        answer = ask(genre)
        st.write("### Answer (From AI Model):")
        st.write(answer)

# -----------------------------
# 5️⃣ Voice Based Recommendation
# -----------------------------
st.subheader("Voice Based Recommendation")

if st.button("🎧 Start Recording"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = r.listen(source)
        st.success("Recording done!")

    st.write("Transcribing audio...")

    try:
        spoken_text = r.recognize_google(audio)
        st.success("You said:")
        st.write(spoken_text)

        # Try CSV filtering first
        filtered_voice = df[df['Genre'].str.contains(spoken_text, case=False, na=False)]

        if not filtered_voice.empty:
            output_voice = ""
            for _, row in filtered_voice.iterrows():
                output_voice += f"- {row['Title']} (Rating: {row['Rating']})\n"

            st.write("### Recommended Movies:")
            st.write(output_voice)

        else:
            st.write("### Generating answer...")
            answer_voice = ask(spoken_text)
            st.write(answer_voice)

    except sr.UnknownValueError:
        st.error("Could not understand audio.")
    except sr.RequestError:
        st.error("Speech API returned an error.")
