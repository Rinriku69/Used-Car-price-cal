## Data source:https://www.kaggle.com/datasets/rupeshraundal/marketcheck-automotive-data-us-canada?select=ca-dealers-used.csv
# 🚗 Used Car Price Predictor

A Machine Learning application designed to estimate the market value of used cars (specifically Honda and Toyota models) based on historical data. This project showcases an **End-to-End ML Pipeline**, from model training to deployment via a web interface.

## 🤖 Machine Learning Workflow
1.  **Data Processing:** Cleans and processes the `honda_toyota_ca.csv` dataset.
2.  **Model Building:** Uses `build_model.py` to train a regression model (e.g., Linear Regression/Random Forest) to understand price correlation with features like year, mileage, and model.
3.  **Model Persistence:** Saves the trained model as `trained_model.joblib` for efficient reuse.
4.  **Deployment:** Uses `price_pred.py` to load the saved model and serve predictions through a Streamlit web app.

## 🛠️ Technologies Used
- **Python**
- **Scikit-Learn**: For building and training the Machine Learning model.
- **Streamlit**: For the frontend user interface.
- **Pandas & NumPy**: For data handling.
- **Joblib**: For saving and loading the trained model.

## 📂 Files Description
- `build_model.py`: Script to train the ML model and save it.
- `price_pred.py`: The main Streamlit application for user interaction.
- `trained_model.joblib`: The pre-trained model file.
- `honda_toyota_ca.csv`: The training dataset.

## 🚀 Usage

### 1. Train the Model (Optional)
If you want to retrain the model with new data:
```bash
python build_model.py
