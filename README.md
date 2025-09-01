# 🧠 Breast Cancer Prediction using Artificial Neural Network

## 📂 About the Dataset

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available through the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).  
It contains features computed from digitized images of fine needle aspirates (FNA) of breast masses.

- **Total Samples:** 569  
- **Classes:** Malignant (M) / Benign (B)  
- **Features:** 30 numerical features (mean, standard error, and worst of cell nuclei measurements)
  
---

## 🔬 Project Overview

This project aims to **predict breast cancer** using a deep learning model built with an **Artificial Neural Network (ANN)**.

### ✅ What I did:
- Performed **Exploratory Data Analysis (EDA)** to explore data distribution and patterns
- Conducted **Feature Engineering**:
  - Feature Scaling using Standardization
  - Handled any null/missing values
  - Encoded labels for classification
- Trained an **ANN model**
- Achieved an **accuracy of ~82%**
- Visualized model performance with **loss and accuracy graphs**

---

## 📊 EDA & Feature Engineering

- Analyzed target class distribution (Malignant vs Benign)
- Created a correlation heatmap to identify strong feature relationships
- Visualized features using pairplots, histograms, and boxplots
- Standardized the features to bring them to a similar scale using `StandardScaler`

---

## 🤖 ANN Architecture

```python
model = Sequential()
model.add(Dense(64, input_shape=(30,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Activation Functions:** ReLU for hidden layers, Sigmoid for output

---

## 📈 Performance

### ✔️ Accuracy Achieved: **~82%**


---

## 🛠️ Tools & Libraries Used

- Python 🐍  
- Pandas, NumPy, Matplotlib, Seaborn 📊  
- TensorFlow / Keras 🤖  
- Scikit-learn  

---

## 💡 Conclusion

This project shows how ANN can be applied to medical datasets to perform binary classification tasks such as breast cancer detection.
