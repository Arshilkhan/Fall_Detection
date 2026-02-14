# Fall Detection Using Deep Learning

Falls among elderly individuals are a major health risk and require immediate detection.
This project proposes an automated fall detection system using deep learning models
trained on image datasets to classify whether a person has fallen or not.

The objective of this project is to compare multiple CNN-based architectures
for accurate fall detection and evaluate their performance using metrics such as
Accuracy, Precision, Recall, F1-score, and ROC-AUC.

The following models were implemented and compared:
- Simple CNN (Custom Architecture)
- ResNet18 (Transfer Learning)
- MobileNetV2 (Lightweight Model)
- EfficientNet-B0 (High Accuracy Model)

Dataset Source: Roboflow Fall Detection Dataset
Classes:
- Fall
- Not Fallen
The dataset contains labeled images split into training, validation, and test sets.

Project Structure:

├── models.py              # Model architectures
├── dataset.py             # Data loading and preprocessing
├── train_utils.py         # Training & evaluation functions
├── run_experiments.py     # Train multiple models
├── batch_test.py          # Run inference on test images
├── requirements.txt       # Dependencies
└── README.md

## Installation

Clone the repository:
git clone https://github.com/yourusername/fall-detection.git
cd fall-detection

Install dependencies:
pip install -r requirements.txt

## Training

To train all models:
python run_experiments.py

## Testing

Place test images inside the test/ folder and run:
python batch_test.py

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve

## Applications:
- Elderly monitoring systems
- Smart hospitals
- Home surveillance safety
- Assisted living environments
