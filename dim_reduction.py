import numpy as np
import pandas as pd
from skmatter.linear_model import PCovR
import matplotlib.pyplot as plt

def simulate_bioreactor_data(n_samples=200, n_sensors=20):
    """Simulates 20 high-dimensional sensor inputs for a 1000L reactor."""
    np.random.seed(42)
    # Latent variables driving the system (e.g., metabolic state)
    latent = np.random.randn(n_samples, 2)
    
    # Sensors are noisy projections of latent states
    sensors = latent @ np.random.rand(2, n_sensors) + np.random.normal(0, 0.1, (n_samples, n_sensors))
    
    # Target: Biomass growth rate (derived from latent state)
    growth_rate = latent[:, 0] * 2 + latent[:, 1] * 0.5 + np.random.normal(0, 0.05, n_samples)
    
    return sensors, growth_rate

def run_pcovr_analysis(X, y):
    """Performs PCovR to find the most important latent drivers of growth."""
    # mixing=0.5 balances feature reconstruction (PCA) and target prediction (Regression)
    pcovr = PCovR(mixing=0.5, n_components=2)
    pcovr.fit(X, y)
    
    X_transformed = pcovr.transform(X)
    
    print("Dimensionality reduction complete.")
    print(f"Top 2 latent components captured from {X.shape[1]} sensors.")
    return X_transformed

if __name__ == "__main__":
    X, y = simulate_bioreactor_data()
    X_red = run_pcovr_analysis(X, y)
    
    # Quick visual check
    plt.scatter(X_red[:, 0], X_red[:, 1], c=y, cmap='viridis')
    plt.colorbar(label='Growth Rate')
    plt.title('Bioreactor States Projected via PCovR')
    plt.xlabel('PCov1')
    plt.ylabel('PCov2')
    plt.show()
