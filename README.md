
# YouTube Video Summarizer

This project is a YouTube Video Summarizer app built using Streamlit. It downloads audio from a YouTube video, transcribes the audio into text, and provides a summary of the transcription in bullet points. The summarization is done using the BART model.

## Features

- Download and convert YouTube video audio into text.
- Summarize the transcription in bullet points.
- Uses the BART model for text summarization.
- Fully functional with Streamlit as the user interface.



## Setup Instructions

1. Clone the Repository

```bash
  git clone https://github.com/ReenaLonare1998/YT_Video_Summarizer

  cd YT_Video_Summarizer
```

2. Create and Activate Virtual Environment

   Create virtual environment

```bash
python -m venv env

```
Activate the virtual environment (Windows)
```bash
env\Scripts\activate
```

Activate the virtual environment (Mac/Linux)
```bash
source env/bin/activate
```

3. Install the Required Packages
All required dependencies are listed in 'requirements.txt' Run the following command to install them:

```bash
pip install -r requirements.txt
```

4. Add the ffmpeg Folder
The ffmpeg folder, which is required to process the audio, can be downloaded from this Google Drive link.
https://drive.google.com/drive/folders/1MI4RS0PAd4bXn1CdaGxgRsLs7Bj_sx4a?usp=drive_link

#### NOTE : Download and place the ffmpeg folder in your project directory (same directory as app.py).

5. Run the Application
Once you have the environment set up and the required files in place, you can start the app using Streamlit:

```bash
streamlit run app.py
```


This will open your default browser and run the application.


## Usage

* Enter the YouTube video URL in the input field.
* Click the Get Summary and Full Transcription button.
* The app will display the summarized transcription in bullet points, followed by the full transcription of the audio.
## Conclusion


This YouTube Video Summarizer app provides a simple and effective way to extract, transcribe, and summarize audio from YouTube videos. By leveraging the power of the Facebook BART model for summarization, users can quickly obtain concise summaries of long video content. The app is easy to set up and run using Streamlit, making it accessible even for those with minimal technical experience. With its efficient and user-friendly design, this project serves as a great tool for content consumption and a practical demonstration of combining various technologies like YouTube-DLP, Pydub, and Hugging Face Transformers in a real-world application.






