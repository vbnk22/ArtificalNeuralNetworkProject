# 🚗 Early Warning System for High Accident Risk on Roads

This project aims to develop an AI-based system that predicts the risk of a road accident based on input data such as traffic conditions, road surface, time of day, and driver-related features. The system is designed to support individual drivers, public safety agencies, fleet operators, and vehicle manufacturers.
---

## 📊 Data

The dataset used is from [Kaggle – Traffic Accident Prediction Dataset](https://www.kaggle.com/datasets/denkuznetz/traffic-accident-prediction/data) and includes:

- Weather conditions (`Weather`)
- Road type (`Road_Type`)
- Time of day (`Time_of_Day`)
- Traffic density (`Traffic_Density`)
- Speed limit (`Speed_Limit`)
- Driver alcohol consumption (`Driver_Alcohol`)
- Driver age and experience (`Driver_Age`, `Driver_Experience`)
- Road surface condition, lighting, vehicle type, accident severity, and more

These features help evaluate accident risk under varying circumstances.

---

## 🧠 Solution Approach

1. **Data Preprocessing:**
   - Filling missing numerical values with median
   - Filling missing categorical values with mode
   - Removing duplicates
   - One-Hot Encoding for categorical features
   - Feature scaling using `StandardScaler`

2. **Modeling:**
   - Binary classification using a **neural network (Keras, TensorFlow)**
   - Architecture:
     - Dense layers with ReLU activation
     - L2 regularization and Dropout to prevent overfitting
     - Output layer with Sigmoid activation

3. **Training & Evaluation:**
   - Binary cross-entropy loss
   - Accuracy metric
   - Achieved ~70% accuracy on the test set

---

## 📈 Results

- The model performs reasonably well, achieving ~70% accuracy.
- Predictions are saved to a generated `out_data.csv` file after each run.
- Limitations:
  - Data does not fully represent real-world conditions
  - Lacks real-time data updates (e.g. weather, traffic incidents)

---

## ✅ Project Goals

The main goal was to create a binary classification model to assess the probability of a road accident based on a given set of features. The project successfully met this goal by developing a neural network that can predict risk with moderate accuracy. With further development and more data, the model could be refined for production use.

---

## 📂 Files

- `main.py` – Contains code for model training and evaluation
- `data.csv` – Input dataset
- `out_data.csv` – Predictions saved after model inference
- `README.md` – Project documentation

---

## 📎 Requirements

- Python 3.x
- pandas, numpy, scikit-learn
- tensorflow, keras, matplotlib

Install with:

```bash
pip install -r requirements.txt
