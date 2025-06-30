# Forest Loss Detection

A deep learning project for detecting and segmenting deforestation areas in satellite/drone imagery using ML segmentation.

## Project Structure

```text
deforestation_segmentation/
├── data/
│   ├── raw/                  # Original images (forest / deforested)
│   ├── processed/            # Train / val splits with masks
│   └── ...
├── models/
│   ├── def_seg_1_main/       # Trained checkpoints (best / latest / final)
│   └── backbones/            # Custom EfficientNet backbone
├── scripts/
│   ├── prepare_data_v2.py    # Advanced data preprocessing (cloud removal, edge enhance)
│   ├── run_deforestation_pipeline.py  # End-to-end data→train pipeline
│   └── ...
├── training/
│   ├── train_unet_b2.py      # Stand-alone training script (UNet++-EffB2)
│   └── ...
├── utils/                    # Helper utilities (visualization, checkpoints, etc.)
├── webapp/
│   ├── app.py                # Streamlit app for inference & analysis
│   └── requirements.txt      # Lightweight deps for the app only
└── config/
    └── hyperparameters.py    # Centralised config dataclasses
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies (core library stack):
```bash
pip install -r requirements.txt
```

Optional – install only the lightweight deps to launch the Streamlit app:
```bash
pip install -r webapp/requirements.txt
```

## Data Preparation

1. Organize your raw data:
   - Place deforestation images in `data/raw/deforested/`
   - Place forest images (including grass areas) in `data/raw/forest/`

2. Prepare the dataset (includes cloud masking & edge refinement):
```bash
python scripts/prepare_data_v2.py \
  --raw_forest_dir data/raw/forest \
  --raw_deforested_dir data/raw/deforested \
  --output_dir data \
  --val_split 0.2
```

## Training

Run the full pipeline (prepare data + train model) with sensible defaults for a 16 GB machine:
```bash
python scripts/run_deforestation_pipeline.py
```

Dacă vrei să rulezi direct scriptul de antrenare UNet++-EffB2:
```bash
python training/train_unet_b2.py --config config/hyperparameters.py
```

## Web Application

1. Run the Streamlit app (after instalarea dependențelor din `webapp/requirements.txt`):
```bash
cd deforestation_segmentation
streamlit run webapp/app.py
```

## Features

- EfficientNet backbone with attention mechanisms
- Segmentation head for deforestation area detection
- Interactive web application for demonstrations
- Comprehensive evaluation metrics
- Visualization tools for predictions

## Model Architecture

The project uses a two-stage approach:
1. Unsupervised pre-training using contrastive learning
2. Supervised fine-tuning for segmentation

## Web Application Features

- Upload satellite/drone images
- Real-time deforestation detection
- Visualization of segmentation masks
- Area calculation of detected deforestation
- Export results and statistics

## Data Organization Tips

1. Image Requirements:
   - Supported formats: .jpg, .png, .tif
   - Images will be resized to 512x512 pixels
   - Maintain consistent image quality and scale

2. Labeling Guidelines:
   - Deforested/: Images showing clear signs of deforestation
   - Forest/: Images of healthy forest areas (can include grass areas)

3. Best Practices:
   - Use high-resolution satellite/drone imagery
   - Ensure good image quality and clarity
   - Include various forest types and conditions
   - Label images consistently

## License

MIT License 
