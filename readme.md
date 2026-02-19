# Transliteration of Handwritten Arabic-Pegon Text into Latin with Convolutional Neural Network (CNN)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Status](https://img.shields.io/badge/Project-Research%20Prototype-green)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to automatically **recognize and transliterate handwritten Arabic-Pegon text into Latin script**.

Arabic-Pegon is an Arabic-derived writing system historically used in Javanese and Indonesian Islamic literature.
This research aims to support **digital preservation, linguistic study, and OCR development for regional scripts** through deep learning–based image recognition.

The system processes handwritten character images through:

1. Image preprocessing
2. CNN-based feature extraction
3. Character classification
4. Transliteration into Latin text

---

## Objectives

* Build an automated recognition system for handwritten Arabic-Pegon characters
* Apply CNN architecture for image-based classification
* Evaluate performance using **accuracy and loss metrics**
* Provide a foundation for further OCR and transliteration research

---

## Key Features

* Handwritten Arabic-Pegon **character recognition**
* **CNN deep learning model** for classification
* Automatic **Latin transliteration output**
* Training & evaluation visualization (accuracy and loss graphs)
* Notebook-based experimentation and extensibility

---

## Tech Stack

* **Python** — core programming language
* **TensorFlow / Keras** — CNN modeling framework
* **OpenCV** — image preprocessing
* **NumPy** — numerical computation
* **Matplotlib** — training visualization

---

## Dataset Description

The dataset contains **handwritten Arabic-Pegon character images** grouped into multiple classes.

### Preprocessing Steps

* Image resizing to uniform resolution
* Grayscale conversion
* Pixel normalization
* Label encoding for classification

These steps ensure **stable training and better convergence** of the CNN model.

---

## Model Architecture

### Convolutional Feature Extraction

The CNN uses stacked convolution blocks:

* **Conv2D** → detect edges, strokes, and shapes
* **ReLU activation** → introduce non-linearity
* **MaxPooling** → reduce spatial size and highlight key features

This allows the model to learn **hierarchical visual representations** of handwritten characters.

---

### Classification Layers

* **Flatten** → convert feature maps into vectors
* **Dense layers** → learn complex feature relationships
* **Dropout** → reduce overfitting
* **Softmax output** → produce probability for each character class

### Architecture Flow

```
Input Image
   ↓
Conv2D → ReLU → MaxPooling
   ↓
Conv2D → ReLU → MaxPooling
   ↓
Flatten
   ↓
Dense → Dropout
   ↓
Softmax Output (Character Class)
```

---

## Training Configuration

Typical experimental parameters:

* **Loss Function:** categorical_crossentropy
* **Optimizer:** Adam
* **Metric:** Accuracy
* **Epochs:** configurable
* **Batch Size:** adjustable

---

## Model Performance

### Accuracy & Loss Monitoring

During training, the system records:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

These metrics help detect:

* Learning progress
* Overfitting / underfitting
* Generalization quality

---

### Training Visualization

Graphs generated using **Matplotlib**:

* **Accuracy vs Epoch**
* **Loss vs Epoch**

**Interpretation example:**

* Increasing accuracy + decreasing loss → good learning
* Large gap between training & validation → overfitting indication

---

## Installation

Clone repository and install dependencies:

```bash
git clone https://github.com/Syahrul-Arh38/Transliteration-of-Handwritten-Arabic-Pegon-Text-into-Latin-with-Convolutional-Neural-Network--CNN-
cd Transliteration-of-Handwritten-Arabic-Pegon-Text-into-Latin-with-Convolutional-Neural-Network--CNN-
pip install tensorflow opencv-python numpy matplotlib
```

---

## Usage

Run the notebook:

```bash
jupyter notebook
```

Then open the main notebook to:

* Train the CNN model
* Evaluate accuracy and loss
* Predict handwritten Arabic-Pegon characters
* Generate Latin transliteration output

---

## Results & Discussion

The CNN model demonstrates the ability to:

* Learn distinguishing **visual stroke patterns**
* Classify handwritten Arabic-Pegon characters with meaningful accuracy
* Provide a **baseline OCR transliteration system** for regional scripts

Performance improvements can be achieved through:

* Larger and more diverse dataset
* Data augmentation techniques
* Deeper CNN or transfer learning
* Hyperparameter tuning

---

## Future Work

Potential extensions of this research:

* Real-time **web or mobile recognition app**
* Full **sentence-level transliteration system**
* Deployment using **TensorFlow Lite**
* Multi-script recognition (Pegon, Javanese, Arabic, etc.)

---

## Author

**Syahrul Ihza Arhamna**
Information Technology Graduate
Focus: **Computer Vision • Deep Learning • Cultural Script Digitization**

---

## License

This project is intended for **educational and research purposes**.
You are free to use, modify, and extend it with proper attribution.
