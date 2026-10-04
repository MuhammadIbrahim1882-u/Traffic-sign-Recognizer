# Traffic Sign Recognizer

A machine learning project that recognizes traffic signs using a CNN trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset. Features a real-time detection system and web dashboard.

## Features

- **Single Sign Recognition**: Classify close-up photos of traffic signs
- **Road Scene Analysis**: Detect and classify multiple signs in street photos
- **Real-time Video**: Process webcam or video file streams
- **Web Dashboard**: Easy-to-use interface with camera support
- **Synthetic Dataset**: Fallback dataset if GTSRB download fails
- **Command-line Tools**: Python scripts for training, prediction, and video analysis

## Project Structure

```
traffic-sign-recognizer/
├── app.py                      # Flask web server
├── train.py                    # Model training script
├── predict.py                  # Single image classification
├── detect_video.py             # Real-time video detection
├── download_data.py            # Dataset downloader
├── generate_synthetic_data.py  # Synthetic data generator
├── make_samples.py             # Sample image/video creator
├── utils.py                    # Shared utilities
├── requirements.txt            # Python dependencies
├── run_all.bat                 # One-click Windows setup
├── README.md                   # This file
├── templates/index.html        # Web dashboard HTML
├── static/
│   ├── style.css              # Dashboard styling
│   └── script.js              # Client-side logic
├── model/                      # Trained models (empty initially)
├── data/                       # Training data (empty initially)
└── samples/                    # Test images/videos (empty initially)
```

## Quick Start

### On Windows

Double-click `run_all.bat`. It will:
1. Create a virtual environment
2. Install dependencies
3. Download/generate training data
4. Train the model (~10-20 minutes on CPU)
5. Open the web app at http://127.0.0.1:5000

### On Linux/Mac

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python train.py
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Usage

### Web Dashboard

1. Click "Choose image" or drag an image onto the canvas
2. Switch between "Single sign" and "Road scene" modes
3. Use your camera with "Use camera" button
4. The model displays its decision and confidence

### Command-line Tools

```bash
# Recognize a single sign
python predict.py photo.jpg

# Real-time webcam detection
python detect_video.py

# Detect signs in a video file
python detect_video.py --source video.mp4 --output result.mp4

# Recognize from a street scene image
python detect_video.py --source street.jpg
```

## Training

```bash
# Auto: use real GTSRB if available, otherwise synthetic
python train.py

# Use specific dataset
python train.py --data gtsrb      # Require real GTSRB
python train.py --data synthetic  # Use synthetic only

# Adjust training
python train.py --epochs 40 --batch 32
```

The model is saved in `model/` along with:
- `traffic_sign_model.keras` - Trained CNN
- `classes.json` - Class mapping
- `training_curves.png` - Loss/accuracy plots
- `confusion_matrix.png` - Classification results
- `metrics.json` - Final statistics

## Dataset

### GTSRB (Real)
Automatically downloaded (~260 MB, 43 classes) from:
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip

Or download manually from Kaggle and extract to `data/gtsrb/`

### Synthetic (Fallback)
Automatically generated if download fails:
- 12 common sign classes
- 300 augmented images per class
- Realistic backgrounds and transformations

## Model

- **Architecture**: CNN with 3 convolutional blocks, batch normalization, dropout
- **Input**: 32×32 RGB images
- **Output**: Softmax over 43 classes + decision level (stop/caution/info/ok)
- **Training**: Adam optimizer, categorical cross-entropy, early stopping
- **Validation accuracy**: ~95%+ on GTSRB

## Sign Detection

Two-stage approach:
1. **Localization**: OpenCV HSV color segmentation (red, blue, yellow) + shape filtering
2. **Classification**: CNN classifies cropped regions

Decision levels:
- **Stop**: Do not proceed (red signs: stop, no entry, etc.)
- **Caution**: Reduce speed, be alert (most yellow and warning signs)
- **Info**: Follow instructions (blue signs: mandatory direction, etc.)
- **Ok**: Restriction ends, normal driving (white/green signs)

## Troubleshooting

### TensorFlow not installing
- Ensure Python 3.10–3.12 (run `python --version`)
- On Windows, use `python -m pip install --upgrade pip` first
- Try `pip install tensorflow-cpu` instead if GPU setup fails

### GTSRB download fails
- Runs automatically create synthetic data as fallback
- Or download manually from Kaggle, extract to `data/gtsrb/`

### Model not training
- Check Python version compatibility (3.10–3.12)
- Ensure all dependencies installed: `pip install -r requirements.txt`
- For GPU support, install tensorflow-gpu and CUDA

### Camera not working in web app
- Use http://127.0.0.1 (not https) or 127.0.0.1:5000
- Requires recent Chrome or Edge
- Browser must have camera permission

## Performance

- **Training**: ~10–20 minutes on CPU, ~2–3 minutes on GPU
- **Inference**: ~50 ms per frame on CPU
- **Web app**: Real-time processing at 20+ fps on modern machines

## License

Educational project. GTSRB dataset has its own licensing; see https://benchmark.ini.rub.de/

## References

- **GTSRB Dataset**: Stallkamp et al., "The German Traffic Sign Recognition Benchmark"
- **TensorFlow/Keras**: https://tensorflow.org
- **OpenCV**: https://opencv.org

---

For questions or issues, check the project folder structure and ensure all directories exist.
