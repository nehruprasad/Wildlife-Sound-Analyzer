import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import json
from scipy.spatial.distance import cdist

st.set_page_config(page_title='Wildlife Sound Intelligence', layout='wide')
st.title('🦜 Wildlife Sound Intelligence')
st.markdown('Record or upload wildlife sounds → identify species → classification + behavior analysis + ecosystem insights')

# ------------------ Helper utilities ------------------

def load_audio(file_bytes, sr=22050):
    try:
        y, _sr = librosa.load(io.BytesIO(file_bytes), sr=sr)
        return y, _sr
    except Exception as e:
        st.error(f'Could not load audio: {e}')
        return None, None


def extract_features(y, sr):
    features = {}
    features['duration'] = float(len(y) / sr)
    features['rms_mean'] = float(np.mean(librosa.feature.rms(y=y)))
    features['zcr_mean'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features['centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

    # Dominant frequency
    S = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    dom_idx = np.argmax(S)
    features['dominant_freq'] = float(freqs[dom_idx])

    # MFCC mean
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(5):
        features[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
    return features, mfcc


# ------------------ Species profiles (mock reference) ------------------
SPECIES_PROFILES = {
    'Indian Peafowl (Peacock)': {'dominant_freq': 1200, 'duration': 2.0, 'rms_mean': 0.02, 'zcr_mean': 0.05},
    'Cicada (Generic Cicada)': {'dominant_freq': 4500, 'duration': 0.5, 'rms_mean': 0.03, 'zcr_mean': 0.08},
    'Frog (Tropical Tree Frog)': {'dominant_freq': 900, 'duration': 0.7, 'rms_mean': 0.015, 'zcr_mean': 0.03},
    'Elephant (Trumpet)': {'dominant_freq': 150, 'duration': 3.0, 'rms_mean': 0.06, 'zcr_mean': 0.02},
    'Lion (Roar)': {'dominant_freq': 250, 'duration': 2.5, 'rms_mean': 0.08, 'zcr_mean': 0.02},
    'Tiger (Roar)': {'dominant_freq': 300, 'duration': 2.2, 'rms_mean': 0.07, 'zcr_mean': 0.025},
    'Wolf (Howl)': {'dominant_freq': 400, 'duration': 3.0, 'rms_mean': 0.05, 'zcr_mean': 0.03},
    'Unknown / Ambiguous': {'dominant_freq': 2000, 'duration': 1.0, 'rms_mean': 0.01, 'zcr_mean': 0.05}
}

SPECIES_VECTOR_KEYS = ['dominant_freq', 'duration', 'rms_mean', 'zcr_mean']


def classify_species(features, k=1, confidence_threshold=0.3):
    vec = np.array([features[kx] for kx in SPECIES_VECTOR_KEYS], dtype=float).reshape(1, -1)
    profiles, names = [], []
    for name, prof in SPECIES_PROFILES.items():
        profiles.append([prof[kx] for kx in SPECIES_VECTOR_KEYS])
        names.append(name)
    profiles = np.array(profiles, dtype=float)
    dists = cdist(vec, profiles, metric='euclidean')[0]
    idx = np.argsort(dists)[:k]

    results = []
    for i in idx:
        score = float(1.0 / (1.0 + dists[i]))
        if score < confidence_threshold:
            results.append({'species': 'Unknown species (low confidence)', 'distance': float(dists[i]), 'confidence': score})
        else:
            results.append({'species': names[i], 'distance': float(dists[i]), 'confidence': score})
    return results


# ------------------ Behavior heuristics ------------------

def behavior_analysis(features):
    behaviors = []
    if features['dominant_freq'] > 3000 and features['duration'] < 1.0:
        behaviors.append('Likely insect chorus or alarm calls.')
    if features['dominant_freq'] < 400 and features['duration'] > 2.0:
        behaviors.append('Low-frequency, long calls — possible large mammal (lion, elephant, tiger, wolf).')
    if features['duration'] < 1.5 and features['zcr_mean'] > 0.04:
        behaviors.append('Short repetitive calls — could indicate bird or amphibian mating/territorial calls.')
    if not behaviors:
        behaviors.append('No strong behavior detected. Further analysis needed.')
    return behaviors


# ------------------ Ecosystem insights ------------------

def ecosystem_insights(detections_list):
    species_seen = {}
    for det in detections_list:
        s = det['species']
        species_seen[s] = species_seen.get(s, 0) + 1
    unique = len(species_seen)
    most_common = max(species_seen.items(), key=lambda x: x[1]) if species_seen else (None, 0)
    return {
        'species_richness_estimate': unique,
        'most_common_species': most_common[0],
        'observations_count': sum(species_seen.values())
    }


# ------------------ Streamlit UI ------------------

st.sidebar.header('Settings')
show_plots = st.sidebar.checkbox('Show plots (waveform, spectrogram, MFCC)', value=True)

uploaded = st.file_uploader('Upload wildlife audio (.wav, .mp3)', type=['wav', 'mp3', 'flac'], accept_multiple_files=True)

if not uploaded:
    st.info('Upload at least one audio file to analyze wildlife sounds.')

all_detections = []

if uploaded:
    for file in uploaded:
        st.header(f'File: {file.name}')
        data = file.read()
        y, sr = load_audio(data, sr=22050)
        if y is None:
            continue
        st.audio(data, format='audio/wav')

        features, mfcc = extract_features(y, sr)
        st.subheader('Extracted features')
        st.json(features)

        detections = classify_species(features, k=2)
        st.subheader('Species classification (nearest-match, mock model)')
        for d in detections:
            st.write(f"{d['species']} — confidence: {d['confidence']:.3f} — distance: {d['distance']:.2f}")
        all_detections.append(detections[0])

        st.subheader('Behavior analysis (heuristic)')
        for b in behavior_analysis(features):
            st.write('-', b)

        if show_plots:
            fig, ax = plt.subplots(3, 1, figsize=(10, 8))
            librosa.display.waveshow(y, sr=sr, ax=ax[0])
            ax[0].set(title='Waveform')
            S = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax[1])
            ax[1].set(title='Spectrogram (dB)')
            librosa.display.specshow(mfcc, x_axis='time', ax=ax[2])
            ax[2].set(title='MFCC')
            fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
            st.pyplot(fig)

        report = {
            'file_name': file.name,
            'features': features,
            'top_detection': detections[0],
            'behavior': behavior_analysis(features)
        }
        st.download_button(f'Download report for {file.name}', data=json.dumps(report, indent=2),
                           file_name=f'{file.name}_report.json', mime='application/json')

if all_detections:
    st.header('Ecosystem insights (batch)')
    insights = ecosystem_insights(all_detections)
    st.json(insights)
    st.markdown(f"**Species richness (estimate):** {insights['species_richness_estimate']}")
    st.markdown(f"**Most common species observed:** {insights['most_common_species']}")

st.markdown('---')
st.caption('Prototype: Uses mock species profiles. For production, train with wildlife sound datasets like ESC-50, Google AudioSet, or custom field recordings.')
