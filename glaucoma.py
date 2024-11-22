import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, 2))
model.load_state_dict(torch.load(r'glaucoma_model.pth', map_location=torch.device('cpu')))
model.eval()

trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

st.title("Glaucoma Detection")
st.write("Upload an eye image to predict whether glaucoma is present or not.")
file=st.file_uploader("Choose a fundus image:", type=["jpg", "jpeg", "png"])
if file is not None:
    img= Image.open(file)
    st.image(img, caption='Uploaded Image',width=150)
    image = trans(img).unsqueeze(0)
    out=model(image)
    x,pred=torch.max(out, 1)
    pre= pred.item()
    classes= ['No Glaucoma', 'Glaucoma']
    st.write(f"Prediction: {classes[pre]}")
