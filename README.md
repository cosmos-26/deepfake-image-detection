# Deepfake Image Detection using Transfer Learning and Fine-Tuning

## Overview

This project focuses on detecting deepfake facial images using Deep Learning and Transfer Learning techniques. The model is built using EfficientNet and fine-tuned on a real vs fake face dataset to classify whether an uploaded image is authentic or manipulated.

The project also includes a Streamlit web application where users can upload an image and receive a prediction indicating whether the image is Real or Fake.

---

# Features

* Deepfake image classification
* Transfer Learning using EfficientNet
* Fine-tuned deep learning model
* Streamlit web application
* Confusion Matrix evaluation
* ROC Curve and AUC analysis
* Real-time image prediction

---

# Technologies Used

* Python
* TensorFlow / Keras
* EfficientNet
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow

---

# Dataset

Dataset used:

* 140K Real and Fake Faces Dataset
* Source: Kaggle

The dataset contains both real and AI-generated facial images used for training and evaluation.

---

# Model Architecture

The project uses EfficientNet as the base model for transfer learning.

Steps followed:

1. Image preprocessing and normalization
2. Transfer Learning using pretrained EfficientNet
3. Fine-tuning selected layers
4. Model training on real and fake face images
5. Performance evaluation using classification metrics

---

# Performance Metrics

Model evaluation results:

* Accuracy: 82%
* Precision: 84%
* Recall: 84%
* F1-Score: 83%
* AUC Score: 0.91

---

# Confusion Matrix

The confusion matrix demonstrates balanced classification performance between real and fake image classes.

![Confusion Matrix](confusion_matrix.png)

---

# ROC Curve

The ROC Curve shows strong discrimination capability between real and fake images.

![ROC Curve](roc_auc_curve.png)

---

# Project Structure

````bash
projectfinal/
│
├── app.py
├── best_model.keras
├── requirements.txt
├── confusion_matrix.png
├── roc_auc_curve.png
└── README.md
---

# Installation

Clone the repository:

```bash
git clone <your-github-repo-link>
```

Move into the project directory:

```bash
cd projectfinal
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Streamlit App

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser at:

```bash
http://localhost:8501
```


# Conclusion

This project demonstrates how transfer learning and fine-tuning can be used for deepfake image detection. The EfficientNet-based model achieved strong classification performance and was successfully integrated into a Streamlit web application for real-time prediction.
