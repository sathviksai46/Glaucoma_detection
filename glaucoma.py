import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# Initialize the model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Modify the final layer
model.load_state_dict(torch.load(r'glaucoma_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict the class of the image
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)  # Forward pass
        _, predicted = torch.max(output, 1)  # Get the predicted class
    return predicted.item()

# Streamlit app layout
st.title("Glaucoma Detection")
st.write("Upload a fundus image")
st.wrtie("heyyyyy")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=False, width=600)

    if st.button("Predict"):
        label = predict(image)
        if label == 0:
            st.write("Prediction: No Glaucoma")
        else:
            st.write("Prediction: Glaucoma")
