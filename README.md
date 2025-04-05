# SmartMeter-CNN-DemographicInference

Using CNN to Predict Demographic Information from Smart Meter Electricity Data  
透過卷積神經網路（CNN）分析智慧電表用電行為，推論使用者的社會人口統計特徵。

---

## Project Overview

This project explores how weekly household electricity usage data, collected from smart meters, can be used to infer demographic characteristics such as age group, employment status, social class, and more. We propose a pure Convolutional Neural Network (CNN) approach that automatically extracts behavioral patterns from the 7×24 electricity matrix.

> 🧠 No manual feature engineering.  
> 📊 No handcrafted rules.  
> Just raw power usage → CNN → demographic prediction.

---

## CNN Architecture

```text
Input: 7x24 weekly electricity matrix
  ↓
[Conv2D] → [ReLU] → [MaxPooling]
  ↓
[Conv2D] → [ReLU] → [MaxPooling]
  ↓
[Conv2D] → [ReLU] → [MaxPooling]
  ↓
[Flatten] → [Dense] → [Dropout]
  ↓
[Softmax Output]
```

- **Input**: One week of hourly power usage (7 days × 24 hours)
- **Conv Layers**: Extract local temporal usage patterns
- **Pooling Layers**: Reduce dimensionality while preserving key signals
- **Dense Layer**: Combine and abstract high-level features
- **Dropout**: Prevent overfitting
- **Softmax**: Output probabilities for classification

---

## 🧹 Data Preprocessing

Raw dataset: `CER Electricity Usage Dataset (2012)`  
Smart meters provide 30-minute interval data → 48 entries per day.

### Preprocessing Steps:

1. **Filter Complete Days**  
   Only keep days with exactly 48 entries (30 min × 48 = 24 hours).

2. **Assemble Weekly Data**  
   - Ensure 7 consecutive days starting from Monday.
   - Resulting in 336 valid entries per household.

3. **Convert to Hourly**  
   - Merge every 2 records → get 24 entries per day.
   - Weekly matrix becomes `7×24`.

4. **Match Labels**  
   - Join with survey-based demographic labels (e.g., age group, retired or not).
   - Remove incomplete or mismatched records.

5. **Save to `.npz`**  
   - Store processed data as `.npz` files for easy loading during training.

---

## 🚀 Training Notes

- **Loss Function**: Categorical Cross-Entropy  
- **Optimizer**: Adam  
- **Metrics**: Accuracy  
- **Validation**: Stratified split on each demographic question  
- **Regularization**: Dropout (0.5) to mitigate overfitting  
- **Batch Size**: 64  
- **Epochs**: 50~100 (depending on early stopping)

---

## 📊 Target Prediction Tasks

We predict multiple user traits, each modeled as a separate classification task:

| Question ID | Demographic Type               | Classes             |
|-------------|--------------------------------|---------------------|
| Q300        | Age group of income earner     | Young / Medium / Old |
| Q310        | Retired or not                 | Yes / No            |
| Q401        | Social class                   | A+B / C1+C2 / D+E   |
| Q410        | Children in household          | Yes / No            |
| Q450        | House type                     | Detached / Semi-detached |
| Q453        | House age                      | Old / New           |
| Q460        | Number of bedrooms             | <3 / 3 / 4 / >4      |
| Q4704       | Cooking facility type          | Electrical / Other  |
| Q4905       | Energy-efficient bulb usage    | Up to half / Half+  |
| Q6103       | House size (continuous)        | (Bucketed)          |

---

## 📁 Project Structure

```
SmartMeter-CNN-DemographicInference/
├── data/
│   ├── raw/              # Original CER txt files
│   ├── processed/        # Weekly 7x24 npz files
│   └── labels/           # Cleaned label .xlsx files
├── model/
│   └── cnn.py            # CNN model architecture
├── train/
│   └── train_model.py    # Training script
├── notebooks/
│   └── exploratory.ipynb # Data exploration & visualization
└── README.md
```

---

## 🧪 Example Use

```python
import numpy as np
from model.cnn import build_model

# Load data
data = np.load('data/processed/Q300_train.npz')
X_train = data['X']
y_train = data['y']

# Build and train model
model = build_model(input_shape=(7, 24, 1), num_classes=3)
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)
```

---

## 📝 Acknowledgements

Data Source: [CER Electricity Usage Dataset (Ireland 2012)]  
Project by: 洪健維、楊秉晟、鍾博安、張晉瑋  
Supervisor: 余金郎 教授，輔仁大學電機工程學系

---

This repository is part of a university research project.  
For academic use only. Please do not reuse without permission.

---
