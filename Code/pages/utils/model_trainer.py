# pages/utils/model_trainer.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.seasonal import MSTL
import warnings
import random
import os
import joblib

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
tf.config.experimental.enable_op_determinism()

DATA_PATH = 'full_stock_data.csv'
MODEL_DIR = 'trained_models'
HISTORY_FILE = os.path.join(MODEL_DIR, 'y_aligned_history.csv')
PERIODS = [143, 687, 3200]

TIME_STEP = 60 
EPOCHS = 30
BATCH_SIZE = 32


def get_model_architecture(key, time_step):
    if key == 'simple':
        model = keras.models.Sequential([
            keras.layers.LSTM(50, input_shape=(time_step, 1)),
            keras.layers.Dense(25, activation="relu"),
            keras.layers.Dense(1)
        ])
    elif key == 'complex':
        model = keras.models.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=(time_step, 1)),
            keras.layers.LSTM(128, return_sequences=False),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1)
        ])
    else: 
        model = keras.models.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1)
        ])
    model.compile(optimizer="adam", loss="mae")
    return model

def train_and_save_lstm(st, component_series, component_name, model_config_key='default'):
    st.write(f"  [LSTM] Training component: {component_name} (using '{model_config_key}' config)")
    scaler = StandardScaler()
    train_data = component_series.values.reshape(-1, 1)
    
    scaler.fit(train_data)
    scaled_data = scaler.transform(train_data)

    X_train, y_train = [], []
    for i in range(TIME_STEP, len(scaled_data)):
        X_train.append(scaled_data[i-TIME_STEP:i, 0])
        y_train.append(scaled_data[i, 0])

    if len(X_train) == 0:
        st.write(f"    ERROR: Not enough data for {component_name}. Skipping.")
        return

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = get_model_architecture(model_config_key, TIME_STEP)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, shuffle=False)

    model.save(os.path.join(MODEL_DIR, f'lstm_{component_name}.keras'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{component_name}.joblib'))
    last_real_sequence = scaled_data[-TIME_STEP:]
    np.save(os.path.join(MODEL_DIR, f'last_sequence_{component_name}.npy'), last_real_sequence)
    st.write(f"    ✓ Saved {component_name} model, scaler, and last sequence.")

def save_seasonal_naive_data(st, component_series, period):
    st.write(f"  [Naive] Saving data for: {component_series.name} (Period: {period})")
    last_cycle = component_series.iloc[-period:]
    last_cycle.to_csv(os.path.join(MODEL_DIR, f'naive_data_p{period}.csv'))
    st.write(f"    ✓ Saved last {period} data points.")

def train_new_models(st_container):
    """
    Runs the full data preparation and model training pipeline.
    Uses st_container (like st or st.status) to write output.
    """
    try:
        st_container.write("--- Starting Data Preparation ---")
        st_container.write(f"Loading {DATA_PATH}...")
        try:
            df = pd.read_csv(DATA_PATH) 
            
            if 'Price' in df.columns:
                st_container.write("Applying legacy CSV cleanup...")
                df = pd.read_csv(DATA_PATH, skiprows=[1, 2])
                df = df.rename(columns={'Price': 'Date'})
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            if df.empty:
                st_container.error("Loaded data file is empty. Cannot train.")
                return
            
            df = df.dropna(subset=['Close'])
            y = df['Close']
            st_container.write("✓ Data loaded and processed.")
        except FileNotFoundError:
            st_container.error(f"ERROR: '{DATA_PATH}' not found. Please run 'Stock Analysis' page first. Cannot train.")
            return
        except Exception as e:
            st_container.error(f"ERROR processing CSV: {e}. Cannot train.")
            return

        st_container.write("Running MSTL decomposition...")
        stl = MSTL(y, periods=PERIODS, iterate=2).fit()
        components_mstl = {
            'Trend': stl.trend,
            'Seasonal_143': stl.seasonal['seasonal_143'],
            'Seasonal_687': stl.seasonal['seasonal_687'],
            'Seasonal_3200': stl.seasonal['seasonal_3200'],
        }
        common_index = stl.trend.dropna().index
        for k in components_mstl:
            components_mstl[k] = components_mstl[k].loc[common_index].dropna()
        resid_series_mstl = stl.resid.loc[common_index].dropna()
        common_index = resid_series_mstl.index
        y_aligned = y.loc[common_index]
        st_container.write("✓ MSTL decomposition done.")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            st_container.write(f"Created directory: {MODEL_DIR}")

        y_aligned.to_csv(HISTORY_FILE)
        st_container.write(f"✓ Aligned historical data saved to '{HISTORY_FILE}'.")
        st_container.write("--- Data Preparation Complete ---")

        st_container.write("-" * 80)
        st_container.write("STARTING MODEL TRAINING PIPELINE")
        
        stl_aligned = MSTL(y_aligned, periods=PERIODS, iterate=2).fit()
        components = {
            'Trend': stl_aligned.trend,
            'Seasonal_143': stl_aligned.seasonal['seasonal_143'],
            'Seasonal_687': stl_aligned.seasonal['seasonal_687'],
            'Seasonal_3200': stl_aligned.seasonal['seasonal_3200'],
        }
        resid_series = stl_aligned.resid
        
        st_container.write("[Main] Training component: Trend (Polynomial Regression, Degree 2)")
        full_data_len = len(y_aligned)
        X_train_lr = np.arange(full_data_len).reshape(-1, 1)
        y_train_lr = components['Trend'].values
        degree = 2
        poly_reg = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
        poly_reg.fit(X_train_lr, y_train_lr)
        joblib.dump(poly_reg, os.path.join(MODEL_DIR, 'trend_model.joblib'))
        st_container.write("  ✓ Trend model trained and saved.")

        st_container.write("[Main] Training component: Seasonal_143")
        train_and_save_lstm(st_container, components['Seasonal_143'], 'Seasonal_143', model_config_key='simple')
        save_seasonal_naive_data(st_container, components['Seasonal_143'], 143)
        st_container.write("  ✓ Seasonal_143 components trained and saved.")

        st_container.write("[Main] Training component: Seasonal_687")
        train_and_save_lstm(st_container, components['Seasonal_687'], 'Seasonal_687', model_config_key='default')
        save_seasonal_naive_data(st_container, components['Seasonal_687'], 687)
        st_container.write("  ✓ Seasonal_687 components trained and saved.")

        st_container.write("[Main] Training component: Seasonal_3200")
        train_and_save_lstm(st_container, components['Seasonal_3200'], 'Seasonal_3200', model_config_key='complex')
        st_container.write("  ✓ Seasonal_3200 model trained and saved.")

        st_container.write("[Main] Training component: Residual")
        train_and_save_lstm(st_container, resid_series, 'Residual', model_config_key='residual')
        st_container.write("  ✓ Residual model trained and saved.")

        st_container.write("-" * 80)
        st_container.write("ALL MODELS TRAINED AND SAVED.")
        st_container.write("--- Training Process Complete ---")
        
    except Exception as e:
        st_container.error(f"An error occurred during training: {e}")