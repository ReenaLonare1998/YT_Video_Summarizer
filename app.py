import streamlit as st
import os
import re
import tempfile
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import speech_recognition as sr
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Move model to CPU
device = torch.device("cpu")
model.to(device)

# Function to download YouTube audio
def download_audio(video_url):
    try:
        # Ensure the temp directory exists
        temp_dir = './temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ffmpeg_location': os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin'),
        }

        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            audio_file = ydl.prepare_filename(info_dict)
            mp3_file = re.sub(r'\.\w+$', '.mp3', audio_file)  # Ensure the correct extension

            if not os.path.isfile(mp3_file):
                mp3_file = audio_file

        return mp3_file

    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None

# Function to convert MP3 to WAV
def convert_to_wav(mp3_file):
    try:
        if mp3_file:
            wav_file = mp3_file.replace('.mp3', '.wav')
            audio = AudioSegment.from_mp3(mp3_file)
            audio.export(wav_file, format="wav")
            return wav_file
        else:
            st.error("No MP3 file to convert.")
            return None
    except Exception as e:
        st.error(f"Error converting MP3 to WAV: {str(e)}")
        return None

# Function to transcribe long audio files in chunks
def transcribe_audio_in_chunks(file_path, chunk_length_ms=30000):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(file_path)
    transcription = ""

    # Process audio in chunks
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = os.path.join('./temp', f"chunk_{i // chunk_length_ms}.wav")
        chunk.export(chunk_path, format="wav")

        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                transcription += text + " "
            except sr.UnknownValueError:
                transcription += "[Unrecognizable] "
            except sr.RequestError as e:
                st.error(f"Error with Google Speech Recognition: {e}")
                return None

        # Optionally delete the chunk file
        os.remove(chunk_path)

    return transcription.strip()

# Function to save transcription to a temporary file
def save_transcription_to_file(transcription_text, temp_dir):
    temp_file_path = os.path.join(temp_dir, "transcription.txt")
    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        temp_file.write(transcription_text)
    return temp_file_path

# Function to load text from a file
def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to summarize a chunk of text
def summarize_chunk(chunk, max_summary_length):
    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_summary_length,
        min_length=int(max_summary_length * 0.5),  # Ensure a reasonable minimum length
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to summarize text based on its length
def summarize_text(text, proportion=0.25, chunk_length=1024):
    inputs = tokenizer(text, return_tensors="pt", max_length=None, truncation=False)
    input_ids = inputs["input_ids"][0]

    # Determine the length of the input text
    total_length = len(input_ids)

    # Calculate summary length based on transcription length
    max_summary_length = int(total_length * proportion)

    # Ensure that summary length is not too small
    max_summary_length = max(max_summary_length, 150)  # Minimum length for summary

    # Summarize the entire text in one go
    return summarize_chunk(text, max_summary_length)

# Function to format summary with bullet points
def format_summary_as_bullet_points(summary):
    bullet_point = "&#8226;"  # HTML entity for a bullet point
    lines = [line.strip().capitalize() for line in summary.split(". ") if line.strip()]
    formatted_summary = "<br>".join(f"{bullet_point} {line}" for line in lines)
    return formatted_summary

# Main function to create the Streamlit app
def main():
    # Set the app's layout and theme
    st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")

    # Custom styling
    st.markdown(
        """
        <style>
        body {
            background-color: #202224;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #202224;
        }
        .title {
            color: #2f557d;
            font-size: 36px;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #5e6f8f;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #4b5a6c;
        }
        .stTextInput > div > input {
            border-radius: 5px;
            border: 1px solid #ced4da;
            padding: 10px;
            font-size: 16px;
        }
        .stAlert {
            background-color: #dbccce;
            color: #721c24;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use HTML for the title to apply custom styles
    st.markdown('<h1 class="title">YouTube Video Summarizer</h1>', unsafe_allow_html=True)

    youtube_url = st.text_input("Enter YouTube video URL:")

    if st.button("Get Summary and Full Transcription"):
        if youtube_url:
            mp3_file = download_audio(youtube_url)
            if mp3_file:
                wav_file = convert_to_wav(mp3_file)
                if wav_file:
                    transcription = transcribe_audio_in_chunks(wav_file)
                    if transcription:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            transcription_file_path = save_transcription_to_file(transcription, temp_dir)
                            transcription_text_from_file = load_text_from_file(transcription_file_path)
                            summary = summarize_text(transcription_text_from_file, proportion=0.25)

                            # Format summary with bullet points
                            formatted_summary = format_summary_as_bullet_points(summary)

                            # Display summary first
                            st.subheader("Summary")
                            st.markdown(f"<div style='width:100%;'>{formatted_summary}</div>", unsafe_allow_html=True)
                           
                            # Display transcription second
                            st.subheader("Transcription")
                            st.markdown(f"<div style='width:100%;'>{transcription_text_from_file}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a YouTube video URL.")

if __name__ == "__main__":
    main()
