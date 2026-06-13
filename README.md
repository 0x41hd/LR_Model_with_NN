# Bike Sharing Demand Prediction using Neural Networks & Linear Regression

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0x41hd/LR_Model_with_NN/blob/main/fcc_Bike_regression.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Deep Learning regression project aimed at predicting bike sharing demand utilizing the **Seoul Bike Sharing Demand Dataset** via **Linear Regression** and **TensorFlow Neural Networks**.

## 📊 Dataset Overview
The dataset contains weather information (Temperature, Humidity, Wind speed, Visibility, Dew point temperature, Solar radiation, Rainfall, Snowfall) and the number of bikes rented per hour.
* **Target Feature:** `bike_count`
* **Special Preprocessing:** Filtered for peak operational hours (`hour == 12`) and optimized using `StandardScaler` for robust model convergence.

## ⚙️ Project Pipeline
1. **Data Cleaning:** Drop non-predictive features (`Date`, `Holiday`, `Seasons`).
2. **Categorical Encoding:** Convert functional status (`Yes`/`No`) into binary format ($1$/$0$).
3. **Feature Scaling:** Normalize weather parameters using `StandardScaler` to handle multi-scale discrepancies.
4. **Modeling:** Comparative performance tracking between classic Scikit-Learn `LinearRegression` and a Multi-Layer Perceptron (MLP) built with `TensorFlow/Keras`.

## 🚀 Getting Started

### Prerequisites
Install all backend compilation dependencies via your terminal:
```bash
pip install -r requirements.txt
