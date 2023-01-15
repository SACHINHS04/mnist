import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms


# Load the trained model's state dictionary
from model import Net

model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

def predict(image):
    # Preprocess the image
    image = Image.open(image)
    image = transforms.ToTensor()(image)
    image = image.view(-1, 1, 28, 28)



    # Pass the image through the model
    output = model(image)
    _, prediction = torch.max(output, 1)

    return prediction

st.set_page_config(page_title="MNIST model", page_icon=":guardsman:", layout="wide")

st.title("MNIST model")

file_image = st.file_uploader("Please upload your image", type=["jpg", "png"])

if file_image:
    image = Image.open(file_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict(file_image)
    st.write("Prediction: ", prediction)
