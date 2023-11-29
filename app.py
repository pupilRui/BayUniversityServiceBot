import streamlit as st
import os
import uuid
from pydub import AudioSegment
from gtts import gTTS
import base64
from cbfs import cbfs
import whisper
from st_audiorec import st_audiorec
import logging  # Import the logging module
import speech_recognition as sr
import io

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, DEBUG, ERROR, etc.)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create a logger for your Streamlit app
logger = logging.getLogger(__name__)

# Sample data for replies (you can replace this with your chatbot logic)
replies = {}
c = cbfs()
audio_model = whisper.load_model("base")

def main():
    st.title("Chatbot App")

    page = st.selectbox("Select a page", ["Text questions", "Voice questions"])
    
    if page == "Text questions":
        display_text_questions_page()
    elif page == "Voice questions":
        display_voice_question_page()

def display_text_questions_page():
    st.header("Chat with the Chatbot")
    question = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if question:
            reply = c.convchain(question)  # Replace with your chatbot logic
            replies[question] = reply
            st.write(f"Question: {question}")
            st.write(f"Reply: {reply}")

            # Log the question and reply
            logger.info("Text Question: %s", question)
            logger.info("Chatbot Reply: %s", reply)

def display_voice_question_page():
    st.header("Ask a Question via Voice")
    st.write("Click the 'Record' button to start recording your question.")
    
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        # st.audio(wav_audio_data, format='audio/wav')

        # Log that audio recording has taken place
        logger.info("Audio Recording Completed")

        # Create a temporary WAV file to store the audio data
        temp_filename = "temp_audio.wav"
        with open(temp_filename, "wb") as temp_audio_file:
            temp_audio_file.write(wav_audio_data)

        result = audio_model.transcribe(temp_filename, language="en")

        # Extract the transcribed text from the result
        question = result["text"]

        # st.write(f"Transcribed Text: {transcription}")

        reply = c.convchain(question)  # Replace with your chatbot logic
        replies[question] = reply
        st.write(f"Question: {question}")
        st.write(f"Reply: {reply}")

        # Log the question and reply
        logger.info("Text Question: %s", question)
        logger.info("Chatbot Reply: %s", reply)

        # Remove the temporary audio file
        os.remove(temp_filename)

        mp3_obj = gTTS(text=reply, lang="en", slow=False)
        reply_filename = f"{uuid.uuid4()}.mp3"
        mp3_obj.save(reply_filename)
        autoplay_audio(reply_filename)

        # # Convert the MP3 content to bytes
        # mp3_bytes_io = io.BytesIO()
        # mp3_obj.write_to_fp(mp3_bytes_io)
        # mp3_bytes = mp3_bytes_io.getvalue()

        # # Encode the MP3 bytes as base64
        # encoded_mp3 = base64.b64encode(mp3_bytes).decode('utf-8')

        # # Play the MP3 audio in Streamlit
        # st.audio(encoded_mp3, format='audio/mp3')

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
