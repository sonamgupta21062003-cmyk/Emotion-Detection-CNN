import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Facial Emotion Detector",
    page_icon="🧠",
    layout="centered"
)

# 2. Cache the model and classes loading so it doesn't reload on every interaction
@st.cache_resource
def load_emotion_model():
    # Load your saved Keras model
    model = tf.keras.models.load_model('final_emotion_model.keras')
    # Load your classes list
    classes = joblib.load('emotion_classes.pkl')
    return model, classes

try:
    model, classes = load_emotion_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or classes: {e}")
    st.info("Please make sure 'final_emotion_model.keras' and 'emotion_classes.pkl' are in the same folder.")
    model_loaded = False

# 3. UI Header layout
st.title("🧠 Facial Emotion Detector")
st.write("Upload an image to predict the human facial expression using your trained CNN model.")

# 4. File Uploader Component
uploaded_file = st.file_uploader("Choose a facial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Convert file stream to PIL Image
    image = Image.open(uploaded_file)
    
    # Create two columns layout to show input and output elegantly
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("📊 Prediction Results")
        
        # --- PREPROCESSING Pipeline matching your Notebook ---
        # Convert PIL Image to numpy array (RGB format)
        img_array = np.array(image.convert('RGB'))
        
        # Resize image to match training input (100x100)
        resized_img = cv2.resize(img_array, (100, 100))
        
        # Add batch dimension -> shape becomes (1, 100, 100, 3)
        input_tensor = np.expand_dims(resized_img, axis=0)
        
        # Normalization if performed during training (uncomment if you scaled features by /255.0)
        # input_tensor = input_tensor / 255.0
        
        # --- INFERENCE ---
        with st.spinner("Analyzing expressions..."):
            predictions = model.predict(input_tensor)[0]
            
        # Get the highest probability index
        predicted_class_idx = np.argmax(predictions)
        predicted_emotion = classes[predicted_class_idx]
        confidence_score = predictions[predicted_class_idx] * 100
        
        # Display Best Result
        st.metric(label="Predicted Emotion", value=predicted_emotion.upper())
        st.progress(int(confidence_score))
        st.write(f"**Confidence Level:** {confidence_score:.2f}%")
        
        # Optional: Show breakdown bar chart of all emotions
        st.write("---")
        st.write("### 📈 Full Probability Breakdown")
        breakdown = dict(zip(classes, predictions))
        st.bar_chart(breakdown)

# 5. Application Footer info
st.markdown("---")
st.caption("Powered by Streamlit & TensorFlow • Trained Data Size: 100x100 px RGB")