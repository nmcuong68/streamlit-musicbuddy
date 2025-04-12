import librosa
import numpy as np
import openai
import wave
import os
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# 🎼 Ước lượng hợp âm từ vector chroma
def estimate_chord(chroma_vector):
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

    chord_templates = {
        "":     [0, 4, 7],        # major
        "m":    [0, 3, 7],        # minor
        "7":    [0, 4, 7, 10],    # dominant 7
        "maj7": [0, 4, 7, 11],    # major 7
        "m7":   [0, 3, 7, 10],    # minor 7
        "dim7": [0, 3, 6, 9]      # diminished 7
    }

    best_score = -1
    best_chord = None

    for i in range(12):
        for suffix, intervals in chord_templates.items():
            template = np.zeros(12)
            for interval in intervals:
                note_index = (i + interval) % 12
                template[note_index] = 1

            score = np.dot(chroma_vector, template)
            if score > best_score:
                best_score = score
                best_chord = pitch_classes[i] + suffix

    return best_chord

# 🎼 Tách hợp âm từ đoạn
def extract_chords_from_frames(y, sr, frame_duration=0.5):
    frame_length = int(sr * frame_duration)
    total_frames = int(len(y) / frame_length)
    chords = []

    for i in range(total_frames):
        start = i * frame_length
        end = start + frame_length
        y_frame = y[start:end]

        if len(y_frame) < frame_length:
            break  # Bỏ qua đoạn cuối nếu không đủ

        # Dùng chroma_cens thay cho chroma_cqt
        chroma = librosa.feature.chroma_cens(y=y_frame, sr=sr)
        avg_chroma = np.median(chroma, axis=1)  # Dùng median thay vì mean

        chord_name = estimate_chord(avg_chroma)
        chords.append(chord_name)

    return chords

# 🔁 Nhận diện vòng hòa âm phổ biến
def detect_common_progressions(chords):
    common_patterns = {
        "I–IV–V": ["C", "F", "G"],
        "vi–IV–I–V": ["Am", "F", "C", "G"],
        "ii–V–I": ["Dm", "G", "C"]
    }

    for name, pattern in common_patterns.items():
        for i in range(len(chords) - len(pattern) + 1):
            if chords[i:i+len(pattern)] == pattern:
                return f"🎵 Phát hiện vòng hòa âm: {name} ({' - '.join(pattern)})"
    return "⚠️ Không phát hiện vòng hòa âm phổ biến."

# 🧠 Gọi GPT để phân tích lời nhạc
def analyze_lyrics(lyrics):
    if not os.getenv("OPENAI_API_KEY"):
        return "❌ Thiếu biến môi trường OPENAI_API_KEY."

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Bạn là một chuyên gia âm nhạc."},
            {"role": "user", "content": f"Phân tích lời bài hát sau và nhận diện chủ đề, cảm xúc và vòng hòa âm nếu có:\n{lyrics}"}
        ],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

# 📊 Vẽ biểu đồ hợp âm
def plot_chord_progression(chords, frame_duration=0.5):
    times = [i * frame_duration for i in range(len(chords))]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(times, [1]*len(chords), width=frame_duration*0.8)

    for i, chord in enumerate(chords):
        ax.text(times[i], 1.05, chord, ha='center', va='bottom', fontsize=10)

    ax.set_yticks([])
    ax.set_xlabel("Thời gian (giây)")
    ax.set_title("🎵 Biểu đồ hợp âm theo thời gian")
    plt.tight_layout()
    return fig