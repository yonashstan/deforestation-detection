import streamlit as st
import warnings
import logging
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import io
import base64
import albumentations as A
import os

# Suppress warnings and configure logging
warnings.filterwarnings('ignore')
logging.getLogger('streamlit.runtime.scriptrunner.magic_funcs').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)

# Configure Streamlit to not watch certain modules
import streamlit.config as config
config.set_option('server.fileWatcherType', 'none')

# --------------------------------------
# CONFIGURATION & PAGE SETUP
# --------------------------------------
APP_TITLE = "Forest Loss Detection - Integrated Model"
APP_ICON = "🌳"
DEFAULT_THRESHOLD = 0.75
MODEL_PATH = Path("deforestation_segmentation/models/def_seg_1_main/best_model.pt")
# Drone imagery parameters
FLIGHT_HEIGHT_M = 100  # Standard drone survey height
GSD_CM_PER_PIXEL = FLIGHT_HEIGHT_M * 2.5 / 100  # Ground sampling distance: ~2.5cm/pixel per 100m height
PIXEL_AREA_M2 = (GSD_CM_PER_PIXEL/100) * (GSD_CM_PER_PIXEL/100)  # Area per pixel in m²

st.set_page_config(
    page_title=APP_TITLE, 
    page_icon=APP_ICON, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add banner image at the top
banner_path = "webapp/images/banner.jpg"
st.markdown('<div class="banner-container">', unsafe_allow_html=True)
try:
    banner_image = Image.open(banner_path)
    st.image(banner_image, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load banner image: {str(e)}")
st.markdown('</div>', unsafe_allow_html=True)

# Inject CSS to center main content
st.markdown(
    """
    <style>
        .banner-container {
            height: 100px;
            overflow: hidden;
            border-radius: 15px;
            margin-bottom: 2rem;
            margin-top: -25rem; /* Pulls the banner up to remove top space */
            gap: -1rem;
        }
        .banner-container img {
            width: 100%;
            height: 10%;
            object-fit: cover;
            margin-bottom: -90px;
        }
        .custom-info-box {
            background-color: #FBFFFA;
            color: #44534A;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .custom-info-box h2, .custom-info-box h3, .custom-info-box li {
            color: #44534A;
        }
        .centered-main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            max-width: 900px;
        }
        .stFileUploader {text-align: center;}
        .stImage {margin: 0 auto; display: block;}
        .column-header {font-size: 1.3rem; font-weight: 600; color: #2E8B57; margin-bottom: 1rem; text-align: center;}
        
        /* Main title styling */
        .main-title {
            margin-top: -4rem;
            font-size: 3.5rem; /* Increased from 2.5rem */
            font-weight: 500;
            text-align: center;
        }
        
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        
        /* Change sidebar header color to green */
        [data-testid="stSidebar"] h2 {
            color: #2E8B57;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------
# HELPER FUNCTIONS
# --------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load segmentation model with cached resource"""
    try:
        # Start with CPU by default to avoid memory issues
        device = torch.device("cpu")
        
        # Create model with EXACT same architecture as v5 training
        model = smp.Unet(
            encoder_name="efficientnet-b0",  # Using B0 encoder as in training
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,  # Handle activation in loss function
            decoder_attention_type="scse",  # Using spatial and channel squeeze-excitation
        )
        
        # Load checkpoint
        ckpt = torch.load(str(path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        # Only try MPS if explicitly enabled and available
        if os.environ.get("USE_MPS", "0") == "1" and torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                model = model.to(device)
                st.success(f"✅ Model loaded successfully on MPS (Apple Silicon GPU)")
            except Exception as e:
                st.warning(f"⚠️ Failed to use MPS, falling back to CPU: {str(e)}")
                device = torch.device("cpu")
        else:
            st.success(f"✅ Model loaded successfully on CPU")
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(img: Image.Image, target_size: tuple = (384, 384)):  # Reduced from 512x512
    """Resize image for inference, normalise and convert to tensor.
    Using a smaller target size to reduce memory usage."""
    try:
        # Resize copy for inference (smaller size to reduce memory)
        img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
        
        # Use simpler transforms to reduce memory
        img_np = np.array(img_resized) / 255.0  # Simple normalization
        
        # Convert to tensor manually (avoid albumentations overhead)
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor, img
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None, None

def predict_mask(model, device, tensor_img):
    """Run a forward pass with careful error handling and memory management."""
    if tensor_img is None:
        return None
        
    def _forward(t_img, dev):
        try:
            with torch.no_grad():
                # Move model to device if needed
                if next(model.parameters()).device != dev:
                    model.to(dev)
                
                # Clear GPU memory before inference
                if dev.type in ['cuda', 'mps']:
                    torch.cuda.empty_cache() if dev.type == 'cuda' else torch.mps.empty_cache()
                
                # Forward pass
                pred = model(t_img.to(dev))
                probs = torch.sigmoid(pred).cpu().numpy()[0, 0]
                
                # Clear memory after inference
                if dev.type in ['cuda', 'mps']:
                    torch.cuda.empty_cache() if dev.type == 'cuda' else torch.mps.empty_cache()
                
                return probs
        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
            return None
    
    try:
        # Try on original device
        result = _forward(tensor_img, device)
        if result is not None:
            return result
            
        # If failed and was using GPU, try CPU
        if device.type in ['cuda', 'mps']:
            st.warning("⚠️ GPU inference failed – retrying on CPU")
            cpu_device = torch.device("cpu")
            return _forward(tensor_img, cpu_device)
            
        return None
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

def mask_to_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4):
    """Overlay semi-transparent red mask on image using the method from test_custom_images.py"""
    # Convert PIL image to numpy array for manipulation
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
    
    # Ensure image is in uint8 format
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Create a colored overlay
    overlay = np.zeros_like(img_np, dtype=np.uint8)
    
    # Boolean representation of the mask
    mask_bool = mask.astype(bool)
    
    # Apply red color to the overlay where the mask is true
    overlay[mask_bool] = [255, 0, 0]  # Red color for deforestation
    
    # Blend the original image and the overlay
    blended = img_np * (1 - alpha) + overlay * alpha
    
    # Clip values to be in the valid range for an image [0, 255]
    return np.clip(blended, 0, 255).astype(np.uint8)

def contours_from_mask(mask: np.ndarray):
    """Extract contours from binary mask"""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(image: np.ndarray, contours, color=(0, 255, 0), thickness=2):
    """Draw contours on image"""
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay

def calculate_deforested_area(mask: np.ndarray, pixel_area_m2: int = PIXEL_AREA_M2):
    """Calculate deforested area in square meters"""
    total_pixels = mask.size
    deforested_pixels = np.sum(mask > 0)
    forest_pixels = total_pixels - deforested_pixels
    
    # Calculate areas
    deforested_area = deforested_pixels * pixel_area_m2
    forest_area = forest_pixels * pixel_area_m2
    total_area = total_pixels * pixel_area_m2
    
    # Calculate percentage (this should remain constant regardless of height)
    deforestation_percentage = (deforested_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    return {
        'deforested_area': deforested_area,
        'forest_area': forest_area,
        'total_area': total_area,
        'deforestation_percentage': deforestation_percentage,
        'deforested_pixels': deforested_pixels,
        'forest_pixels': forest_pixels,
        'total_pixels': total_pixels
    }

def create_analysis_report(image_name: str, mask: np.ndarray, threshold: float, stats: dict):
    """Create a comprehensive analysis report"""
    report = f"""
# Forest Loss Detection Report

## Analysis Summary
- **Image Analyzed**: {image_name}
- **Analysis Date**: {st.session_state.get('analysis_date', 'N/A')}
- **Model Used**: Deforestation Segmentation (EfficientNet-B2 + U-Net)
- **Confidence Threshold**: {threshold:.2f}
- **Flight Height**: {st.session_state.get('flight_height', FLIGHT_HEIGHT_M)} meters
- **Ground Resolution**: {st.session_state.get('flight_height', FLIGHT_HEIGHT_M) * 2.5 / 100:.1f} cm/pixel

## Area Statistics
- **Total Area Analyzed**: {stats['total_area']:.1f} m²
- **Deforested Area**: {stats['deforested_area']:.1f} m²
- **Forest Area**: {stats['forest_area']:.1f} m²
- **Deforestation Percentage**: {stats['deforestation_percentage']:.1f}%

## Pixel Statistics
- **Total Pixels**: {stats['total_pixels']:,}
- **Deforested Pixels**: {stats['deforested_pixels']:,}
- **Forest Pixels**: {stats['forest_pixels']:,}

## Risk Assessment
"""
    
    if stats['deforestation_percentage'] < 5:
        report += "- **Risk Level**: Low - Minimal deforestation detected\n"
    elif stats['deforestation_percentage'] < 15:
        report += "- **Risk Level**: Moderate - Some deforestation detected\n"
    elif stats['deforestation_percentage'] < 30:
        report += "- **Risk Level**: High - Significant deforestation detected\n"
    else:
        report += "- **Risk Level**: Critical - Extensive deforestation detected\n"
    
    return report

def plot_statistics(stats: dict):
    """Create visualization charts with error handling for negative values"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart (using pixel counts to ensure correct proportions)
        labels = ['Forest', 'Deforested']
        sizes = [stats['forest_pixels'], stats['deforested_pixels']]
        
        # Check if we have any valid data
        if sum(sizes) <= 0:
            st.warning("⚠️ No valid data to display in charts")
            return None
            
        colors = ['#228B22', '#DC143C']
        
        # Create pie chart with percentage labels
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Area Distribution')
        
        # Bar chart (using actual areas)
        categories = ['Forest', 'Deforested']
        values = [stats['forest_area'], stats['deforested_area']]
        
        bars = ax2.bar(categories, values, color=colors)
        ax2.set_title('Area by Category (m²)')
        ax2.set_ylabel('Area (m²)')
        
        # Add value labels on bars with better formatting
        for bar, value in zip(bars, values):
            if value >= 1000:
                value_text = f'{value:,.0f}'
            else:
                value_text = f'{value:.1f}'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    value_text, ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")
        return None

def add_border_to_image(image: np.ndarray, border_color: list, border_width: int = 5):
    """Add a colored border to an image"""
    if len(image.shape) == 2:  # Grayscale image
        height, width = image.shape
        bordered_image = np.zeros((height + 2 * border_width, width + 2 * border_width), dtype=image.dtype)
        bordered_image[:border_width, :] = border_color[0]  # Use first color channel
        bordered_image[-border_width:, :] = border_color[0]
        bordered_image[:, :border_width] = border_color[0]
        bordered_image[:, -border_width:] = border_color[0]
        bordered_image[border_width:border_width+height, border_width:border_width+width] = image
    else:  # RGB image
        height, width = image.shape[:2]
        bordered_image = np.zeros((height + 2 * border_width, width + 2 * border_width, 3), dtype=image.dtype)
        bordered_image[:border_width, :, :] = border_color
        bordered_image[-border_width:, :, :] = border_color
        bordered_image[:, :border_width, :] = border_color
        bordered_image[:, -border_width:, :] = border_color
        bordered_image[border_width:border_width+height, border_width:border_width+width, :] = image
    
    return bordered_image

def update_pixel_area(height_m):
    """Update pixel area based on flight height"""
    gsd = height_m * 2.5 / 100  # cm/pixel
    return (gsd/100) * (gsd/100)  # Convert to m² per pixel

# --------------------------------------
# MAIN APP LAYOUT
# --------------------------------------

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Analysis Settings")
    
    # Flight height adjustment
    flight_height = st.slider(
        "Drone Flight Height (m)", 
        50, 200, FLIGHT_HEIGHT_M, 10,
        help="Adjust the drone flight height to calculate correct ground resolution"
    )
    
    # Update pixel area based on height
    pixel_area = update_pixel_area(flight_height)
    # Store in session state for use in calculations
    st.session_state['flight_height'] = flight_height
    st.session_state['pixel_area'] = pixel_area
    st.caption(f"Ground resolution: {flight_height * 2.5 / 100:.1f} cm/pixel")
    st.caption(f"Pixel area: {pixel_area:.6f} m²")
    
    # Model selection
    model_option = st.selectbox(
        "Model Version",
        [
            # "Best 2 (Synth)",
            # "Last 2 (Synth)",
            # "Best 3 (Synth v3)",
            # "Last 3 (Synth v3)",
            "Best Fast v2"  # Added new model option
        ],
        help="Choose which trained model to use for segmentation"
    )
    
    # Update model path based on selection
    if model_option == "Best 2 (Synth)":
        MODEL_PATH = Path("models/deforestation_segmentation_synth/deforestation_model_best.pth")
    elif model_option == "Last 2 (Synth)":
        MODEL_PATH = Path("models/deforestation_segmentation_synth/deforestation_model_latest.pth")
    elif model_option == "Best 3 (Synth v3)":
        MODEL_PATH = Path("models/deforestation_segmentation_synth_v3/deforestation_model_best.pth")
    elif model_option == "Last 3 (Synth v3)":
        MODEL_PATH = Path("models/deforestation_segmentation_synth_v3/deforestation_model_latest.pth")
    elif model_option == "Best Fast v5":
        MODEL_PATH = Path("v5/models/deforestation_fast2/best_model.pt")
    
    # Analysis parameters
    threshold = st.slider(
        "Detection Threshold", 
        0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
        help="Higher threshold = more conservative detection"
    )
    
    view_option = st.selectbox(
        "Visualization Mode", 
        ["Overlay", "Contours", "Both", "Mask Only"],
        index=2,
        help="Choose how to visualize the detection results"
    )
    
    alpha = st.slider(
        "Overlay Transparency", 
        0.1, 0.9, 0.4, 0.1,
        help="Adjust the transparency of the deforestation overlay"
    )
    
    # Display options
    st.header("📊 Display Options")
    show_stats = st.checkbox("Show Statistics", value=True)
    show_charts = st.checkbox("Show Charts", value=True)
    show_report = st.checkbox("Generate Report", value=True)

# Centered main content
with st.container():
    st.markdown('<div class="centered-main">', unsafe_allow_html=True)
    st.markdown(f"<div class='main-title'>{APP_ICON} {APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Upload satellite or drone imagery to detect and analyze forest loss patterns using advanced deep learning segmentation</div>",
        unsafe_allow_html=True
    )
    st.header("Image Analysis")
    st.markdown('<div class="column-header">📤 Input Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a forest image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload high-resolution imagery for best results"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    # Store analysis date
    from datetime import datetime
    st.session_state['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display original image
    img_pil = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="column-header">Original Image</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)
    
    # Load model and process
    with st.spinner("🔄 Loading model and analyzing image..."):
        model, device = load_model(MODEL_PATH)
        
        if model is not None:
            # Preprocess image
            tensor_img, img_original = preprocess_image(img_pil)
            
            # Get prediction
            probabilities = predict_mask(model, device, tensor_img)
            binary_mask = (probabilities > threshold).astype(np.uint8)
            
            # Ensure mask has same dimensions as original image
            if binary_mask.shape != img_original.size[::-1]:  # PIL size is (width, height), numpy is (height, width)
                # Resize mask to match original image
                mask_pil = Image.fromarray(binary_mask)
                mask_pil = mask_pil.resize(img_original.size, Image.Resampling.NEAREST)
                binary_mask = np.array(mask_pil)
            
            # Prepare visualizations
            img_np = np.array(img_original)
            
            # Create different visualization options
            if view_option == "Overlay":
                result_img = mask_to_overlay(img_original, binary_mask, alpha)
                caption = "Deforestation Overlay"
            elif view_option == "Contours":
                contours = contours_from_mask(binary_mask)
                result_img = draw_contours(img_np, contours)
                caption = "Deforestation Contours"
            elif view_option == "Both":
                contours = contours_from_mask(binary_mask)
                outline_img = draw_contours(img_np, contours)
                result_img = mask_to_overlay(outline_img, binary_mask, alpha)
                caption = "Overlay + Contours"
            else:  # Mask Only
                result_img = (binary_mask * 255).astype(np.uint8)
                caption = "Binary Mask"
            
            # Add green border to result image
            result_img_with_border = add_border_to_image(result_img, border_color=[0, 255, 0], border_width=8)
            
            # Display result with green border
            with col2:
                st.markdown('<div class="column-header">🎯 Analysis Result</div>', unsafe_allow_html=True)
                st.image(result_img_with_border, caption=f"{caption} (Result)", use_container_width=True)
            
            # Calculate statistics
            stats = calculate_deforested_area(binary_mask, pixel_area)
            
            # Display statistics
            if show_stats:
                st.header("📊 Analysis Results")
                
                # Create metric columns
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "Deforested Area", 
                        f"{stats['deforested_area']:,.1f} m²",
                        help="Area detected as deforested"
                    )
                
                with metric_col2:
                    st.metric(
                        "Forest Area", 
                        f"{stats['forest_area']:,.1f} m²",
                        help="Area detected as forest"
                    )
                
                with metric_col3:
                    st.metric(
                        "Deforestation %", 
                        f"{stats['deforestation_percentage']:.1f}%",
                        help="Percentage of area deforested"
                    )
                
                with metric_col4:
                    st.metric(
                        "Total Area", 
                        f"{stats['total_area']:,.1f} m²",
                        help="Total area analyzed"
                    )
                
                # Progress bar for deforestation percentage
                st.progress(
                    min(int(stats['deforestation_percentage']), 100) / 100, 
                    text=f"Deforestation Level: {stats['deforestation_percentage']:.1f}%"
                )
            
            # Show charts
            if show_charts:
                st.header("📈 Visualizations")
                fig = plot_statistics(stats)
                if fig is not None:
                    st.pyplot(fig)
            
            # Generate and display report
            if show_report:
                st.header("Analysis Report")
                report = create_analysis_report(
                    uploaded.name, binary_mask, threshold, 
                    stats
                )
                st.markdown(report)
            
            # Download section
            st.header("Download Results")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                if st.button("📄 Download Report"):
                    report_text = create_analysis_report(
                        uploaded.name, binary_mask, threshold, 
                        stats
                    )
                    st.download_button(
                        label="📄 Save Report",
                        data=report_text,
                        file_name=f"deforestation_report_{uploaded.name.split('.')[0]}.md",
                        mime="text/markdown"
                    )
            
            with download_col2:
                if st.button("🖼️ Download Mask"):
                    # Save binary mask
                    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
                    buffer = io.BytesIO()
                    mask_img.save(buffer, format='PNG')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="🖼️ Save Mask",
                        data=buffer.getvalue(),
                        file_name=f"deforestation_mask_{uploaded.name.split('.')[0]}.png",
                        mime="image/png"
                    )
            
            with download_col3:
                if st.button("🖼️ Download Overlay"):
                    # Save overlay image using the original image
                    overlay_img = mask_to_overlay(img_original, binary_mask, alpha)
                    overlay_pil = Image.fromarray(overlay_img)
                    buffer = io.BytesIO()
                    overlay_pil.save(buffer, format='PNG')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="🖼️ Save Overlay",
                        data=buffer.getvalue(),
                        file_name=f"deforestation_overlay_{uploaded.name.split('.')[0]}.png",
                        mime="image/png"
                    )

else:
    # Show welcome message and instructions with custom styling
    info_html = """
    <div class="custom-info-box">
        <h2>Getting Started</h2>
        <p>
            <b>Upload an Image</b>: Use the file uploader to select a satellite or drone image<br>
            <b>Adjust Settings</b>: Use the sidebar to configure detection parameters<br>
            <b>Analyze</b>: The model will automatically process your image<br>
            <b>Review Results</b>: View statistics, charts, and download results
        </p>
        <h3>📋 Supported Formats</h3>
        <ul>
            <li>PNG, JPG, JPEG images</li>
            <li>High-resolution imagery recommended</li>
            <li>RGB color space</li>
        </ul>
        <h3>🔧 Model Information</h3>
        <ul>
            <li><b>Architecture</b>: U-Net with EfficientNet-B0 encoder</li>
            <li><b>Training</b>: Synthetic deforestation dataset</li>
            <li><b>Resolution</b>: 30m per pixel (Landsat-like)</li>
        </ul>
    </div>
    """
    # Place custom info box inside the centered container
    with st.container():
        st.markdown('<div class="centered-main">', unsafe_allow_html=True)
        st.markdown(info_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        🌳 Forest Loss Detection System | Powered by Deep Learning | Stan Yonash | 2025 |
        <a href='#' style='color: #2E8B57; text-decoration: none;'>Documentation</a>
    </div>
    """,
    unsafe_allow_html=True
) 