import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
import tempfile
import wave
import librosa
import os

from music_buddy.audio_tools import (
    extract_chords_from_frames,
    detect_common_progressions,
    plot_chord_progression,
    analyze_lyrics
)

st.set_page_config(page_title="🎵 Music Buddy", layout="centered")
st.title(":musical_note: Music Buddy - Phân tích hợp âm từ file âm thanh hoặc ghi âm trực tiếp")

FRAME_DURATION = 1.0

st.sidebar.header("🔢 Chọn cách nhập âm thanh")
mode = st.sidebar.radio("Nguồn âm thanh", ["Upload file WAV", "Ghi âm trực tiếp"])

audio_file = None
y = None
sr = None

if mode == "Upload file WAV":
    uploaded_file = st.file_uploader("📂 Tải lên file .wav", type=["wav"])
    if uploaded_file is not None:
        audio_file = uploaded_file
        y, sr = librosa.load(uploaded_file, sr=None)

elif mode == "Ghi âm trực tiếp":
    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.frames = []

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray().flatten().astype(np.int16)
            self.frames.append(pcm)
            return frame

    ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioProcessor,
    )

    if ctx.audio_processor and not ctx.state.playing and ctx.audio_processor.frames:
        audio_data = np.concatenate(ctx.audio_processor.frames).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_file = f.name
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_data.tobytes())

        y, sr = librosa.load(audio_file, sr=None)

# Phân tích hợp âm khi đã có file âm thanh
if y is not None and sr is not None:
    chords = extract_chords_from_frames(y, sr)

    st.subheader("🎶 Chuỗi hợp âm:")
    st.write(" → " + " - ".join(chords))

    st.subheader("📊 Biểu đồ hợp âm:")
    fig = plot_chord_progression(chords, frame_duration=FRAME_DURATION)
    st.pyplot(fig)

    st.subheader("🔀 Vòng hoà âm phổ biến:")
    st.success(detect_common_progressions(chords))

    st.subheader("📝 Phân tích lời bài hát bằng GPT")
    lyrics = st.text_area("Nhập lời bài hát tiếng Anh hoặc Việt:")
    if st.button("🧐 GPT phân tích"):
        if lyrics.strip():
            with st.spinner("Đang phân tích..."):
                gpt_output = analyze_lyrics(lyrics)
                st.write(gpt_output)
        else:
            st.warning("⚠️ Vui lòng nhập lời bài hát.")