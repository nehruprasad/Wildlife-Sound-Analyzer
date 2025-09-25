import streamlit as st
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# ----------------- Helper Functions -----------------

# Dummy species classifier for demonstration
SPECIES_LIST = ["Sparrow", "Crow", "Wolf", "Elephant", "Frog", "Unknown"]

def classify_species(audio_path):
    """
    Dummy species classification: returns random species for demo purposes
    Replace with ML model inference in production.
    """
    species = np.random.choice(SPECIES_LIST)
    confidence = round(np.random.uniform(0.6, 0.99), 2)
    return species, confidence

def behavior_analysis(species):
    """
    Simple behavior analysis based on species
    """
    behavior_dict = {
        "Sparrow": "Chirping, Nesting",
        "Crow": "Cawing, Alert Calls",
        "Wolf": "Howling, Pack Communication",
        "Elephant": "Trumpeting, Social Interaction",
        "Frog": "Croaking, Mating Calls",
        "Unknown": "Unknown behavior"
    }
    return behavior_dict.get(species, "Unknown behavior")

def ecosystem_insights(species_list):
    """
    Generate ecosystem insights based on detected species
    """
    insights = []
    if "Wolf" in species_list:
        insights.append("Presence of predators in area")
    if "Elephant" in species_list:
        insights.append("Large herbivores present; indicates forest health")
    if "Sparrow" in species_list or "Crow" in species_list:
        insights.append("Bird species active; indicates daytime activity")
    if "Frog" in species_list:
        insights.append("Wetland or water body nearby")
    if not insights:
        insights.append("No notable ecosystem indicators")
    return insights

def plot_audio_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=(8,2))
    plt.title("Audio Waveform")
    plt.plot(y)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# ----------------- Streamlit App -----------------

st.set_page_config(page_title="Wildlife Sound Analyzer", layout="wide")
st.title("🦉 Wildlife Sound Analyzer (Audio Classification)")

st.markdown("Record or upload wildlife sounds to identify species, analyze behavior, and provide ecosystem insights.")

# Upload or record audio
uploaded_file = st.file_uploader("Upload wildlife audio (WAV/MP3/FLAC)", type=["wav","mp3","flac"])

if uploaded_file:
    # Save temporary file
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        audio_path = tmp_file.name
    
    st.success("Audio uploaded successfully!")

    # Display audio player
    st.audio(audio_path)

    # Plot waveform
    st.subheader("📊 Audio Waveform")
    plot_audio_waveform(audio_path)

    # Species classification
    st.subheader("🦜 Species Classification")
    species, confidence = classify_species(audio_path)
    st.write(f"Detected Species: **{species}** with confidence: {confidence*100}%")

    # Behavior analysis
    st.subheader("🔍 Behavior Analysis")
    behavior = behavior_analysis(species)
    st.write(f"Observed Behavior: **{behavior}**")

    # Ecosystem insights
    st.subheader("🌿 Ecosystem Insights")
    insights = ecosystem_insights([species])
    for insight in insights:
        st.write("- " + insight)
