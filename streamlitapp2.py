import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load the trained model
model_path = r"C:\Users\samar\potato_plant_disease\model_neural"  # Specify the actual path to your saved model
model = tf.keras.models.load_model(model_path)

# Define the class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Replace with your actual class names

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    return img

# Function to display the image with clamped values
def display_image(img):
    img = np.clip(img, 0.0, 1.0)  # Clamp pixel values to [0, 1]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")

    # Convert the plot to a NumPy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    img_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img_array

# Streamlit app
st.title("Potato Leaf Disease Detection")

# Upload an image through the Streamlit interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the uploaded image
    image = preprocess_image(uploaded_file)

    # Display the uploaded image
    st.image(display_image(tf.squeeze(image, axis=0).numpy()), caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for model prediction
    input_image = image / 255.0  # Normalize pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(input_image)
    predicted_class = class_names[np.argmax(predictions[0])]

    # Display the predicted class
    st.subheader("Prediction:")
    st.write(f"The model predicts: {predicted_class}")


