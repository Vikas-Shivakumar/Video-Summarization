import streamlit as st
import os
import json
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import subprocess
from transformers import pipeline
import cv2
import easyocr
import shlex

# Load SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure directories exist
os.makedirs("uploaded_videos", exist_ok=True)
os.makedirs("summaries", exist_ok=True)

# **Custom CSS for Dark Mode & Stylish Buttons**
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #6200EE;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# **Sidebar Navigation**
st.sidebar.title("🔍 Navigation")
option = st.sidebar.radio("Go to", ["Home", "Upload Video", "Q&A"])

# Load the summarization model
summarizer = pipeline("summarization", model="t5-small")

# **Step 2: Extract Audio**
def extract_audio(video_path, output_audio="output_audio.wav"):
    os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 "{output_audio}"')
    return output_audio

# **Step 3: Transcribe Audio**
def transcribe_audio(audio_path, output_json="transcription.json"):
    model = WhisperModel("small")
    segments, _ = model.transcribe(audio_path)

    transcript_data = {"transcription": []}
    full_transcription = ""
    for segment in segments:
        text = segment.text.strip()
        transcript_data["transcription"].append({
            "start_time": round(segment.start, 2),
            "end_time": round(segment.end, 2),
            "text": text
        })
        full_transcription += f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}\n"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4)

    return output_json, full_transcription

# **Step 4: Segment Transcript**
def segment_transcript(json_file, output_json="segmented_transcript.json", words_per_segment=100):
    with open(json_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    sentences = [segment["text"] for segment in transcript_data["transcription"]]
    timestamps = [(segment["start_time"], segment["end_time"]) for segment in transcript_data["transcription"]]

    segmented_data = []
    current_segment = []
    current_start_time = timestamps[0][0]
    word_count = 0

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        word_count += len(words)
        current_segment.append(sentence)

        if word_count >= words_per_segment or i == len(sentences) - 1:
            segment_text = " ".join(current_segment)
            segment_end_time = timestamps[i][1]

            segmented_data.append({
                "start_time": round(current_start_time, 2),
                "end_time": round(segment_end_time, 2),
                "content": segment_text
            })

            if i < len(sentences) - 1:
                current_start_time = timestamps[i + 1][0]
            current_segment = []
            word_count = 0

    transcript_data["segments"] = segmented_data
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=4)

    return output_json, segmented_data

def create_faiss_index(json_file, faiss_index="faiss_transcript.index", metadata_file="faiss_metadata.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    segment_texts = [segment["content"] for segment in transcript_data["segments"]]
    embeddings = sbert_model.encode(segment_texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, faiss_index)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(transcript_data["segments"], f, indent=4)

def summarize_text(text, max_length=150):
    """Summarizes the given text using a Transformer model."""
    if len(text) < 50:
        return text  # No need to summarize very short text

    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def translate_text(text, target_language):
    """Translates text into the target language using Google Translator."""
    translator = GoogleTranslator(source="auto", target=target_language)
    return translator.translate(text)

# **📜 Generate `.srt` Subtitles**
def generate_srt(transcript_data, output_srt):
    """Converts transcription data to SRT format."""
    srt_content = ""
    counter = 1

    for segment in transcript_data["transcription"]:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        text = segment["text"]

        def format_time(seconds):
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)},{str(millis).zfill(3)}"

        start_srt = format_time(start_time)
        end_srt = format_time(end_time)

        srt_content += f"{counter}\n{start_srt} --> {end_srt}\n{text}\n\n"
        counter += 1

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return output_srt

# **🌍 Translate Subtitles**
def translate_srt(input_srt, target_language):
    """Translates subtitles using Google Translator API."""
    translated_srt = input_srt.replace(".srt", f"_{target_language}.srt")

    with open(input_srt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_content = ""
    for line in lines:
        if "-->" in line or line.strip().isdigit():
            translated_content += line  # Keep timestamps & numbering unchanged
        elif line.strip():
            translated_content += GoogleTranslator(source="auto", target=target_language).translate(line.strip()) + "\n"
        else:
            translated_content += "\n"

    with open(translated_srt, "w", encoding="utf-8") as f:
        f.write(translated_content)

    return translated_srt

# **🔥 Burn Subtitles into Video Using FFmpeg**
def add_subtitles_to_video(video_path, srt_path, output_video="video_with_subtitles.mp4"):
    """Burns subtitles into video using FFmpeg."""

    # Convert paths to absolute & ensure forward slashes
    video_path = os.path.abspath(video_path).replace("\\", "/")
    srt_path = srt_path
    output_video = os.path.abspath(output_video).replace("\\", "/")
    print(video_path, srt_path, output_video)

    try:
        command = f'ffmpeg -i "{video_path}" -vf "subtitles={srt_path}" -c:a copy "{output_video}"'
        print(f"🔹 Running FFmpeg Command: {command}")  # Debugging

        process = subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        print(f"✅ FFmpeg Output: {process.stdout.decode()}")

        return output_video
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg Error: {e.stderr.decode()}") #display error in streamlit
        print("🚨 FFmpeg Error:", e.stderr.decode())
        return None
    except FileNotFoundError:
        st.error("FFmpeg not found. Please ensure FFmpeg is installed and in your PATH.")
        return None

def extract_keyframes(video_path, output_folder, frame_interval=30):
    """Extracts keyframes from a video at every `frame_interval` seconds."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keyframes = []
    for i in range(0, frame_count, frame_interval * fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            keyframes.append((frame_path, i // fps))  # Store frame path and timestamp (seconds)

    cap.release()
    return keyframes

reader = easyocr.Reader(['en'])  # Load OCR model for English

def extract_text_from_frames(frames):
    """Extracts text from a list of keyframes using OCR."""
    frame_texts = {}

    for frame_path, timestamp in frames:
        text = reader.readtext(frame_path, detail=0)  # Extract text from image
        frame_texts[timestamp] = " ".join(text)  # Store extracted text

    return frame_texts

def upload_video():
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        video_path = os.path.join("uploaded_videos", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Video uploaded: {uploaded_file.name}")
        st.video(video_path)
        return video_path
    return None

def clear_session_state():
    keys_to_clear = ["audio_path", "full_transcription", "segmented_data",
                     "faiss_loaded", "faiss_index", "segments", "srt_path",
                     "keyframes", "extracted_texts", "summary", "qa_results"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

if option == "Home":
    st.subheader("🏠 Welcome to Educational Video AI!")
    st.write("A smart AI-powered video assistant to help you understand educational videos better.")

elif option == "Upload Video":
    video_path = upload_video()

    if video_path:
        clear_session_state()
        progress = st.progress(0)

        # Initial video processing (run only once on upload)
        with st.spinner("⏳ Processing video..."):
            audio_path = extract_audio(video_path)
            time.sleep(3)
            progress.progress(10)
            st.session_state["audio_path"] = audio_path

            transcription_json, full_transcription = transcribe_audio(audio_path)
            time.sleep(10)
            progress.progress(25)
            st.session_state["full_transcription"] = full_transcription

            segmented_json, segmented_data = segment_transcript(transcription_json)
            time.sleep(5)
            progress.progress(40)
            create_faiss_index(segmented_json)
            time.sleep(2)
            progress.progress(50)
            st.session_state["segmented_data"] = segmented_data
            st.session_state["faiss_loaded"] = True
            st.session_state["faiss_index"] = faiss.read_index("faiss_transcript.index")
            st.session_state["segments"] = segmented_data

            with open("transcription.json", "r", encoding="utf-8") as f:
                transcription_data = json.load(f)
            srt_path = os.path.join("subtitles_en.srt")
            generate_srt(transcription_data, srt_path)
            st.session_state["srt_path"] = srt_path
            progress.progress(60)

            keyframes = extract_keyframes(video_path, "keyframes")
            if keyframes:
                extracted_texts = extract_text_from_frames(keyframes)
                st.session_state["keyframes"] = keyframes
                st.session_state["extracted_texts"] = extracted_texts
            progress.progress(75)

            st.session_state["summary"] = summarize_text(full_transcription)
            progress.progress(90)

        progress.progress(100)
        st.success("✅ Video processing complete!")

        # Display initial outputs
        if "audio_path" in st.session_state:
            with st.expander("🎵 **Audio File**", expanded=True):
                st.audio(st.session_state["audio_path"])

        if "full_transcription" in st.session_state:
            with st.expander("📜 **Full Transcription**", expanded=True):
                st.text_area("Transcribed Text", st.session_state["full_transcription"], height=300, key="transcript_box")

        if "segmented_data" in st.session_state:
            with st.expander("📌 **Segmented Transcript**", expanded=True):
                for i, segment in enumerate(st.session_state["segmented_data"]):
                    st.markdown(f"""
                        <div style='border:1px solid #ddd; padding:10px; border-radius:10px; margin:10px 0; padding:10px; background-color:#f5f5f5;'>
                            <h4>📌 Segment {i+1}</h4>
                            <p>🕒 <b>{segment['start_time']}s - {segment['end_time']}s</b></p>
                            <p>{segment['content']}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # Subtitle Translation and Video Playback (Language Change Handling)
        st.subheader("🎬 Subtitles")
        language = st.selectbox("🌍 Choose Subtitle Language", ["English (Default)", "Spanish", "French", "German", "Hindi", "Chinese"], key="subtitle_lang")
        if "srt_path" in st.session_state:
            output_video = os.path.join("uploaded_videos", f"subtitled_{language.replace(' ', '_')}.mp4")

            if language != "English (Default)":
                lang_code = language[:2].lower()
                translated_srt_path = st.session_state["srt_path"].replace(".srt", f"_{lang_code}.srt")

                if not os.path.exists(translated_srt_path):
                    with st.spinner(f"Translating subtitles to {language}..."): #added spinner
                        translated_srt = translate_srt(st.session_state["srt_path"], lang_code)
                else:
                    translated_srt = translated_srt_path

                if not os.path.exists(output_video):
                    with st.spinner(f"Burning subtitles to video for {language}..."): #added spinner
                        add_subtitles_to_video(video_path, translated_srt_path, output_video)

                if os.path.exists(output_video):
                    st.video(output_video)
                else:
                    st.error("Error creating subtitled video.")
            else:
                output_video = os.path.join("uploaded_videos", f"subtitled_en.mp4")

                if not os.path.exists(output_video):
                    with st.spinner("Burning English subtitles to video..."):add_subtitles_to_video(video_path, st.session_state["srt_path"], output_video)

                if os.path.exists(output_video):
                    st.video(output_video)
                else:
                    st.error("Error creating subtitled video.")

        # Frame Information Display
        if "keyframes" in st.session_state:
            st.subheader("📌 Keyframes (Click to Jump)")
            for frame_path, timestamp in st.session_state["keyframes"]:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(frame_path, caption=f"🕒 {timestamp}s", use_container_width=True)
                with col2:
                    text = st.session_state["extracted_texts"].get(timestamp, "No text detected")
                    st.markdown(f"**Extracted Text:** {text}")
                    if st.button(f"▶️ Jump to {timestamp}s", key=f"jump_{timestamp}"):
                        st.session_state["video_time"] = timestamp
                        st.rerun()

        # Summarization and Translation Display (Language Change Handling)
        st.subheader("📜 Summarization")
        st.text_area("📌 Summarized Text", st.session_state["summary"], height=100)

        language_sum = st.selectbox("🌍 Choose Summary Language", ["Spanish", "French", "German", "Hindi", "Chinese"], key="summary_lang")
        if "summary" in st.session_state:
            lang_code_sum = language_sum[:2].lower()
            output_summary_file = os.path.join("summaries", f"summary_{language_sum}.txt")

            if not os.path.exists(output_summary_file):
                with st.spinner(f"Translating summary to {language_sum}..."): #added spinner
                    translated_summary = translate_text(st.session_state["summary"], lang_code_sum)
                    with open(output_summary_file, "w", encoding="utf-8") as f:
                        f.write(translated_summary)

            with open(output_summary_file, "r", encoding="utf-8") as f:
                st.text_area(f"📌 Translated Summary ({language_sum})", f.read(), height=100)

elif option == "Q&A":
    st.subheader("🔍 Ask a Question")
    question = st.text_input("Enter your question:")

    if st.button("🤖 Get Answer"):
        if "faiss_loaded" in st.session_state and st.session_state["faiss_loaded"]:
            index = st.session_state["faiss_index"]
            segments = st.session_state["segments"]
            question_embedding = sbert_model.encode([question])
            D, I = index.search(np.array(question_embedding, dtype=np.float32), k=3)

            st.session_state["qa_results"] = [segments[idx] for idx in I[0] if idx < len(segments)]

    if "qa_results" in st.session_state:
        st.subheader("📌 Relevant Answers")
        for segment in st.session_state["qa_results"]:
            st.markdown(f"""
                <div style='border:1px solid #ddd; padding:10px; border-radius:10px; margin:10px 0; padding:10px; background-color:#f5f5f5;'>
                    <h4>💡 Answer</h4>
                    <p>🕒 <b>{segment['start_time']}s - {segment['end_time']}s</b></p>
                    <p>{segment['content']}</p>
                </div>
            """, unsafe_allow_html=True)