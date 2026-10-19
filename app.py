import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from tensorflow.keras.models import load_model

# PAGE CONFIG
st.set_page_config(
    page_title="Skin Disease Classification Dashboard",
    page_icon="🩺",
    layout="wide"
)
# CUSTOM CSS

st.markdown("""
<style>

.main{
    background-color:#f5f9ff;
}

.header-box{
    background: linear-gradient(90deg,#0f4c81,#2d7dd2);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.header-box h1{
    font-size:65px;
    margin-bottom:10px;
}

.header-box h4{
    font-size:28px;
}

.prediction-card{
    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 4px 20px rgba(0,0,0,0.15);
    text-align:center;
}

.prediction-card h1{
    color:#0f4c81;
    font-size:48px;
}

.prediction-card h2{
    font-size:30px;
}

.info-card{
    background:#eef6ff;
    padding:20px;
    border-radius:15px;
    font-size:20px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# HEADER

st.markdown("""
<div class="header-box">
<h1>🩺 Skin Disease Classification System</h1>
<h4>AI Powered Skin Lesion Detection Using Deep Learning</h4>
</div>
""", unsafe_allow_html=True)

# DASHBOARD CARDS

c1, c2, c3, c4 = st.columns(4)

c1.metric("📸 Dataset Images", "10,015")
c2.metric("🦠 Classes", "7")
c3.metric("🤖 Model", "CNN")
c4.metric("📐 Input Size", "96×96")
st.markdown("---")

# SIDEBAR

st.sidebar.title("📋 Project Information")

st.sidebar.info("""
Dataset: HAM10000

Technology:
• TensorFlow
• Keras
• Streamlit

Model:
• CNN

Classes:
• AKIEC
• BCC
• BKL
• DF
• MEL
• NV
• VASC
""")

# LOAD MODEL

@st.cache_resource
def load_my_model():
    return load_model("skin_disease_model.h5")

model = load_my_model()

# LOAD LABEL ENCODER

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# DISEASE NAMES

disease_names = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

# DISEASE INFO

disease_info = {
    "akiec": "A precancerous skin lesion caused by prolonged sun exposure.",
    "bcc": "The most common type of skin cancer and usually grows slowly.",
    "bkl": "A benign skin lesion that is generally non-cancerous.",
    "df": "A harmless fibrous skin growth.",
    "mel": "A serious and potentially dangerous form of skin cancer.",
    "nv": "A common mole that is generally benign.",
    "vasc": "A lesion associated with blood vessels."
}

# IMAGE UPLOAD

st.subheader("📤 Upload Skin Lesion Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# PREDICTION

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = image.resize((96, 96))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("🔍 Analyzing Image..."):

        prediction = model.predict(img)

        predicted_index = np.argmax(prediction)

        predicted_class = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(np.max(prediction) * 100)

    with col2:

        st.markdown(f"""
        <div class="prediction-card">
        <h2>Prediction Result</h2>
        <h1>{disease_names[predicted_class]}</h1>
        <h2>Confidence: {confidence:.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.subheader("🎯 Confidence Level")
        st.progress(int(confidence))

        # Risk Assessment
        high_risk = ["mel", "bcc", "akiec"]

        st.subheader("🚨 Risk Assessment")

        if predicted_class in high_risk:
            st.error("🔴 High Risk - Medical Evaluation Recommended")
        else:
            st.success("🟢 Low Risk - Likely Benign")

        st.subheader("📖 Disease Information")

        st.markdown(f"""
        <div class="info-card">
        {disease_info[predicted_class]}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🩺 Recommendation")

        if predicted_class in high_risk:
            st.warning("""
Consult a dermatologist for further evaluation.

Early diagnosis and treatment are important.
""")
        else:
            st.info("""
The lesion appears benign according to the model.

Regular monitoring is recommended.
""")

    st.markdown("---")

    # Top Predictions

    st.subheader("🏆 Top 3 Predictions")

    top3_idx = np.argsort(prediction[0])[-3:][::-1]

    for idx in top3_idx:
        disease = label_encoder.inverse_transform([idx])[0]
        prob = prediction[0][idx] * 100

        st.write(f"**{disease.upper()}** — {prob:.2f}%")

    st.markdown("---")

    # Probability Chart

    st.subheader("📊 Prediction Distribution")

    prob_df = pd.DataFrame({
        "Disease": label_encoder.classes_,
        "Probability (%)": prediction[0] * 100
    })

    st.bar_chart(
        prob_df.set_index("Disease")
    )

    st.dataframe(
        prob_df,
        use_container_width=True
    )


# FOOTER

st.markdown("---")

st.warning(
    "⚠️ This application is intended for educational and research purposes only and should not replace professional medical diagnosis."
)

st.markdown("""
<div class="footer">
<h3>Skin Disease Classification using HAM10000 Dataset</h3>
Developed using TensorFlow, Keras and Streamlit
</div>
""", unsafe_allow_html=True)