import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import time

# ————————————————————————————————————————————————
# 1. SIGNAL PREPROCESSING (Week 8 Design)
# ————————————————————————————————————————————————

def apply_bandpass(data, fs, low=0.5, high=40.0):
    """Digital Filter Stage: Removes noise/artifacts."""
    nyq = 0.5 * fs
    b, a = butter(5, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, data, axis=0)

# ————————————————————————————————————————————————
# 2. HYBRID AI MODEL (Week 7/8 Roadmap)
# ————————————————————————————————————————————————

def build_octavision_model(input_shape, num_classes):
    """
    Implements CNN + LSTM hybrid architecture.
    CNN extracts spatial features; LSTM extracts temporal patterns.
    """
    model = models.Sequential([
        # Section 2: Feature Learning (CNN)
        layers.Conv1D(64, kernel_size=12, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(), 
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=6, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # Section: Temporal Learning (LSTM)
        layers.LSTM(64, return_sequences=False),
        
        # Section 3: Feature Consolidation
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), 
        
        # Section 4: Prediction (Softmax)
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ————————————————————————————————————————————————
# 3. DATA MANAGEMENT & DISCOVERY
# ————————————————————————————————————————————————

def load_and_inspect(file):
    mat = loadmat(file)
    eeg = mat.get('cnt')
    nfo = mat.get('nfo')
    mrk = mat.get('mrk')
    
    try:
        fs = float(nfo[0][0]['fs'][0][0])
    except:
        fs = 100.0
        
    discovery = {"fs": fs, "markers": None}
    if mrk is not None:
        y_labels = mrk[0][0]['y'][0]
        pos = mrk[0][0]['pos'][0]
        valid_mask = ~np.isnan(y_labels)
        discovery["markers"] = {
            "unique": np.unique(y_labels[valid_mask]),
            "raw_y": y_labels,
            "pos": pos
        }
    return eeg, discovery

# ————————————————————————————————————————————————
# 4. STREAMLIT INTERFACE
# ————————————————————————————————————————————————

st.set_page_config(page_title="OctaVision Hybrid Decoder", layout="wide")
st.sidebar.title("🧠 System Control")
uploaded = st.sidebar.file_uploader("Upload EEG .mat file", type="mat")

if not uploaded:
    st.title("OctaVision: Brain-Signal Communication")
    st.info("Upload dataset to initialize the CNN+LSTM hybrid pipeline.")
    st.stop()

eeg, info = load_and_inspect(uploaded)

tab0, tab1, tab2, tab3 = st.tabs(["📊 Project Dashboard", "🔎 Discovery", "🚀 Training", "⚡ Live Decoder"])

# --- TAB 0: PROJECT DASHBOARD (UPDATED) ---
with tab0:
    st.title("OctaVision: Research Archive")
    st.markdown("### Bridging the gap between trapped thoughts and digital expression.")
    
    m1, m2 = st.columns(2)
    m1.metric("AI Architecture", "CNN + LSTM", "Hybrid Pipeline")
    m2.metric("Status", "Phase 3 Ready", "Week 10")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📘 Project Mission")
        st.write("""
        OctaVision is an EEG-based Brain-Computer Interface developed using Design Thinking 
        to help non-speaking individuals (ALS, Stroke, Paralysis) communicate. 
        The system bypasses physical speech production by decoding motor imagery 
        directly from the motor cortex.
        """)
        
        st.subheader("👥 Development Team")
        team = [
            "Anchuri Harshith Sai", "Janit Pradyun", "Tejsai.ch", 
            "Kautilya Ramisetti", "Rishi Kamishetty", "Siddharth.M", 
            "Anshu Vanguru", "Rithya Reddy"
        ]
        st.write(", ".join(team))

    with col2:
        st.subheader("📂 Dataset Deep-Dive")
        st.info("""
        **Dataset:** BCI Competition IV (Dataset IVa)
        
        **About the Data:** This dataset was recorded by Fraunhofer FIRST (Berlin) and consists of EEG signals from 5 healthy subjects. 
        The specific subject used in this pipeline ('aa') provided high-resolution signals across 118 channels.
        
        **Motor Imagery Task:** Instead of actual physical movement, the subject was instructed to **imagine** performing a task. 
        The AI scans for neural markers in the motor cortex associated with:
        * **Imagined Right Hand movement** (Decoded as **YES**)
        * **Imagined Right Foot movement** (Decoded as **NO**)
        
        **Recording Specs:** - 118 EEG channels (scanned over the motor cortex).
        - 100 Hz sampling frequency.
        - Non-invasive scalp electrodes.
        """)

# --- TAB 1: DISCOVERY (Untouched) ---
with tab1:
    st.header("1. Neural Marker Discovery")
    if info['markers'] is not None:
        labels = info['markers']['unique']
        st.write("**Detected Classes:**", labels)
        
        col1, col2 = st.columns(2)
        yes_id = col1.selectbox("Select 'YES' ID", labels, index=0)
        no_id = col2.selectbox("Select 'NO' ID", labels, index=1 if len(labels)>1 else 0)
        st.session_state['mapping'] = {yes_id: "YES", no_id: "NO"}
    else:
        st.error("No 'mrk' structure found in this file.")

# --- TAB 2: TRAINING (Untouched) ---
with tab2:
    st.header("2. Hybrid Model Development")
    if 'mapping' in st.session_state:
        if st.button("Start Hybrid CNN+LSTM Training"):
            pos, y = info['markers']['pos'], info['markers']['raw_y']
            fs = info['fs']
            window = int(2 * fs)
            
            X, Y = [], []
            scaler = StandardScaler()
            
            with st.spinner("Extracting multi-channel features..."):
                for i in range(len(pos)):
                    p = pos[i]
                    if (p + window < len(eeg)) and (not np.isnan(y[i])):
                        segment = eeg[p:p+window, :10] 
                        segment = apply_bandpass(segment, fs)
                        segment = scaler.fit_transform(segment)
                        X.append(segment)
                        Y.append(y[i])
            
            if len(X) > 0:
                X, Y = np.array(X), np.array(Y)
                le = LabelEncoder()
                Y_enc = le.fit_transform(Y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, Y_enc, test_size=0.2)
                
                model = build_octavision_model((X.shape[1], X.shape[2]), len(le.classes_))
                
                monitor = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
                
                history = model.fit(X_train, y_train, epochs=30, batch_size=16, 
                                    validation_data=(X_test, y_test), callbacks=[monitor], verbose=0)
                
                st.session_state['final_model'] = model
                st.session_state['encoder'] = le
                st.session_state['scaler'] = scaler
                
                val_acc = history.history['val_accuracy'][-1]
                st.metric("Hybrid System Accuracy", f"{val_acc*100:.1f}%")
                st.success("Decoder Optimized with LSTM temporal learning.")
            else:
                st.error("Dataset insufficient for deep learning.")

# --- TAB 3: LIVE DECODER (Untouched) ---
with tab3:
    st.header("3. Real-Time Predictive Intent")
    if 'final_model' in st.session_state:
        if st.button("Activate Live Decoding"):
            box = st.empty()
            model = st.session_state['final_model']
            le, sc = st.session_state['encoder'], st.session_state['scaler']
            mapping = st.session_state['mapping']
            
            for start in range(0, len(eeg) - int(info['fs']*2), int(info['fs']*0.5)):
                chunk = eeg[start : start + int(info['fs']*2), :10]
                chunk = apply_bandpass(chunk, info['fs'])
                chunk = sc.transform(chunk)
                
                input_tensor = np.expand_dims(chunk, axis=0)
                probs = model.predict(input_tensor, verbose=0)
                pred_idx = np.argmax(probs)
                
                original_id = le.inverse_transform([pred_idx])[0]
                result = mapping.get(original_id, "IDLE")
                confidence = np.max(probs) * 100
                
                color = "green" if result == "YES" else "red"
                
                with box.container():
                    st.markdown(f"""
                        <div style="text-align: center; border: 10px solid {color}; padding: 40px; border-radius: 20px;">
                            <h1 style="font-size: 100px; color: {color};">{result}</h1>
                            <h3>Confidence: {confidence:.1f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                time.sleep(0.1)