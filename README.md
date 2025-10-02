# 🚦 Traffic Sign Classification – Task 4 (Elevvo Internship)

This project is the **final task** of my internship at **Elevvo** 🎉.  
It focuses on building a **deep learning model** for **traffic sign classification** using **TensorFlow/Keras**.

---

## 📌 Project Overview
The system classifies traffic signs into multiple categories using a **Convolutional Neural Network (CNN)**.  
The workflow includes:
- Dataset loading from CSV files (`Train.csv`, `Test.csv`, `Meta.csv`)
- Preprocessing: resizing to **32x32 RGB**, normalization, one-hot encoding
- Building a custom **CNN model** with Conv2D, BatchNorm, MaxPooling, Dense & Dropout
- Training & evaluation with performance metrics and plots

---

## 🛠 Tech Stack
- **Python 3.11**
- **TensorFlow / Keras**
- **OpenCV** (image processing)
- **Matplotlib & Seaborn** (visualizations)
- **Scikit-learn** (classification report & confusion matrix)

---

## ⚙️ Model Architecture
- **Conv2D(32) + BatchNorm + MaxPooling**
- **Conv2D(64) + BatchNorm + MaxPooling**
- **Conv2D(128) + BatchNorm + MaxPooling**
- **Flatten → Dense(256, ReLU) → Dropout(0.5)**
- **Dense(num_classes, Softmax)**

Optimizer: **Adam**  
Loss Function: **Categorical Crossentropy**  
Epochs: **10**  
Batch Size: **64**

---

## 📊 Results
- ✅ **Classification Report** (precision, recall, F1-score)  
- ✅ **Confusion Matrix** visualization  
- ✅ **Accuracy & Loss curves**  

### Sample Plots (replace with your actual images):
![Training Accuracy](results/accuracy_plot.png)
![Training Loss](results/loss_plot.png)
![Confusion Matrix](results/confusion_matrix.png)

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-classification.git
   cd traffic-sign-classification
