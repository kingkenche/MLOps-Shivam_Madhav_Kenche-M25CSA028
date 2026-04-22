import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="CityScape Image Segmentation", layout="wide")

# Define functions for prediction
@st.cache_resource
def load_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=23,
    )
    if os.path.exists("unet_model.pth"):
        model.load_state_dict(torch.load("unet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_mask(model, image_pil):
    # Resize and prepare image
    img = image_pil.resize((128, 128), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).numpy()
    return pred

def colorize_mask(mask, num_classes=23):
    # Simple colormap
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    # Background to black
    colors[0] = [0, 0, 0]
    
    colored_mask = colors[mask]
    return Image.fromarray(colored_mask.astype('uint8'))

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Page 1: Training Metrics", "Page 2: Inference App"])

    if page == "Page 1: Training Metrics":
        st.title("Training Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Loss Curve")
            if os.path.exists("Question2/training_loss.png"):
                st.image("Question2/training_loss.png", use_container_width=True)
            else:
                st.warning("Training loss plot not found.")
                
        with col2:
            st.subheader("mIOU & mDice Scores")
            if os.path.exists("Question2/metrics.png"):
                st.image("Question2/metrics.png", use_container_width=True)
            else:
                st.warning("Metrics plot not found.")
                
        st.subheader("Test Set Results")
        if os.path.exists("Question2/test_metrics.txt"):
            with open("Question2/test_metrics.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    st.write(f"**{line.strip()}**")
        else:
            st.warning("Test metrics not found.")

    elif page == "Page 2: Inference App":
        st.title("Inference on Test Images")
        st.write("Upload up to 4 input images from the test set to see the ground truth and predictions.")
        
        uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        
        if uploaded_files:
            if len(uploaded_files) > 4:
                st.warning("Please upload a maximum of 4 images. Showing the first 4.")
                uploaded_files = uploaded_files[:4]
                
            model = load_model()
            
            for file in uploaded_files:
                st.subheader(f"Results for: {file.name}")
                col1, col2, col3 = st.columns(3)
                
                # Input Image
                image_pil = Image.open(file).convert("RGB")
                with col1:
                    st.write("**Input Image**")
                    st.image(image_pil, use_container_width=True)
                
                # Ground Truth
                mask_path = os.path.join("data", "CameraMask", file.name)
                with col2:
                    st.write("**Ground Truth Mask**")
                    if os.path.exists(mask_path):
                        gt_mask = Image.open(mask_path)
                        gt_mask_arr = np.array(gt_mask)[:, :, 0]
                        gt_colored = colorize_mask(gt_mask_arr)
                        st.image(gt_colored, use_container_width=True)
                    else:
                        st.write("Ground truth not available for this image.")
                        
                # Prediction
                with col3:
                    st.write("**Predicted Mask**")
                    pred_mask = predict_mask(model, image_pil)
                    # Resize prediction to match original image size for display
                    pred_mask_img = Image.fromarray(pred_mask.astype('uint8')).resize(image_pil.size, Image.NEAREST)
                    pred_colored = colorize_mask(np.array(pred_mask_img))
                    st.image(pred_colored, use_container_width=True)

if __name__ == "__main__":
    main()
