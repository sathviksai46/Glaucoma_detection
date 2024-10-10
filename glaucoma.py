import streamlit as st
import torch
from torchvision import transforms
from torchvision import models
from PIL import Image

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Sequential(torch.nn.Dropout(p=0.5), torch.nn.Linear(512, 2))
model.load_state_dict(torch.load(r'glaucoma_model.pth', map_location=torch.device('cpu')))
model.eval()

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def classify(img):
    img = img.unsqueeze(0)
    out=model(img)
    x,pred=torch.max(out.data, 1)
    return pred.item()

st.title("Glaucoma Detection")
st.write("Upload an image for glaucoma detection.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img= Image.open(uploaded_file)
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying.")
    img=trans(img)
    pred=classify(img)
    c= {0: "No Glaucoma", 1: "Glaucoma"}
    st.write("Prediction:", c[pred])
