# ☕ Coffee Price Forecasting

This project focuses on forecasting coffee prices using a **hybrid time series modeling approach**.
The model decomposes the price series into multiple components and applies the most suitable forecasting technique to each part.

The goal is to improve prediction accuracy for **nonlinear, volatile commodity prices** like coffee.

---

## 📌 Project Overview

Coffee prices:

* Show strong nonlinear trends
* Contain multiple long-term cycles
* Exhibit high volatility

Instead of using a single model, this project:

1. Decomposes the time series using **MSTL**.
2. Models each component separately.
3. Recombines the predictions into the final forecast.

This approach captures both:

* Long-term trend behavior
* Short-term fluctuations

---

## 🧠 Hybrid Model Design

The series is split into:

* Trend component
* Seasonal cycle (143 days)
* Seasonal cycle (687 days)
* Seasonal cycle (3200 days)
* Residual component

### Models per component

| Component     | Model Used                       |
| ------------- | -------------------------------- |
| Trend         | Polynomial Regression (degree 2) |
| Seasonal 143  | Naive + LSTM (ensemble)          |
| Seasonal 687  | Naive + LSTM (ensemble)          |
| Seasonal 3200 | LSTM                             |
| Residual      | LSTM                             |

Final forecast = sum of all component predictions.

---

## 📊 Final Results

Test data size: **648 samples**

| Metric | Value |
| ------ | ----- |
| RMSE   | 12.26 |
| MAE    | 8.35  |
| MAPE   | 2.90% |

---

## 📂 Project Structure

```
Coffee-Price-Forecasting/
│
├── Code/
│   Contains all scripts used for:
│   - Data preprocessing
│   - Model training
│   - Forecasting
│   - Benchmark comparisons
│
├── Documents/
│   - Final project report
│   - Presentation slides
│
├── Latex/
│   - Source files used to generate the article
│   - Overleaf-compatible project files
│
├── Result/
│   Contains experimental outputs:
│   - Model performance metrics
│   - Model comparison results
│   - Seasonality and trend strength analysis
│   - Evidence supporting results in the article
│
└── README.md
└── procedure.txt
└── requirements.txt

```

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* Statsmodels (MSTL, ARIMA)
* Scikit-learn
* NumPy, Pandas, Matplotlib


---

## 🚀 How to Run the Application

Make sure:

* You have **Python installed**
* Required libraries are installed
* You are **connected to the internet**

---

### 1. Go to the project code directory

```bash
cd Code
```

---

### 2. Start the Streamlit application

```bash
streamlit run trading_app.py
```

The app will automatically open in your browser at:

```
http://localhost:8501
```

If it does not open automatically, manually open the link in your browser.

---

### 3. Train the models

1. In the Streamlit sidebar, go to:
   **“Stock Prediction”**
2. Click:
   **“Train Initial Models”**
   or
   **“Train on updated data”**
3. Wait for the training process to complete.

---

### 4. Use the application

Once training finishes:

* The models are ready to use.
* You can explore predictions and features inside the app.
* Most functions are automatic—just wait for processing when prompted.

---

### Install dependencies (if not already installed)

From the `Code` directory:

```bash
pip install -r requirements.txt
```

---

## 📚 Benchmarked Models

The hybrid model is compared with:

* ARIMA
* LSTM
* Extreme Learning Machine (ELM)
* Polynomial + Seasonal Naive baseline

---

## 👨‍💻 Authors

* Adarsh S. Kamatagi – 23BCS005
* Rohan Babhale Laxmikant – 23BCS026

Course: **Statistics for Computer Science (CS309)**


---

## 📜 License

This project is for academic and research purposes only.



