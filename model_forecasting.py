import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def generate_process_data(n_points=1000):
    """Generates synthetic sensor data for a carbon capture system."""
    np.random.seed(42)
    temp = np.random.normal(310, 5, n_points)      # Temperature in K
    flow_rate = np.random.normal(50, 10, n_points) # CO2 flow rate
    
    # Target: pH (influenced by flow and temp)
    ph = 10 - (flow_rate * 0.05) + np.random.normal(0, 0.1, n_points)
    
    return pd.DataFrame({'temp': temp, 'flow_rate': flow_rate, 'ph': ph})

def train_forecaster(df):
    X = df[['temp', 'flow_rate']]
    y = df['ph']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(f"Model Training Complete. MAE: {mean_absolute_error(y_test, predictions):.4f}")
    return model

if __name__ == "__main__":
    data = generate_process_data()
    train_forecaster(data)
