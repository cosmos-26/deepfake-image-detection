import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
model = load_model("best_model.keras")

# Title
st.title("Deepfake Image Detection")

st.write("Upload an image to detect whether it is Real or Fake.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Show prediction score
    st.write("Prediction Score:", round(float(prediction), 2))

    # Threshold logic
    if prediction > 0.75:
        st.success("Real Image Detected")

    elif prediction < 0.25:
        st.error("Fake Image Detected")

    else:
        st.warning("Uncertain Prediction")