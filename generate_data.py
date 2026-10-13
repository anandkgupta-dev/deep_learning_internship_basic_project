"""
generate_data.py
================
Generates a realistic, reproducible synthetic educational dataset for
weather condition classification using Temperature and Humidity.

DISCLAIMER:
This dataset is generated for educational demonstrations and college coursework.
It does NOT represent real meteorological observations, and temperature and humidity
alone are insufficient to reliably forecast actual weather conditions.
"""

import os
import numpy as np
import pandas as pd

def generate_weather_dataset(num_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic weather observations for Sunny, Cloudy, and Rainy conditions.
    
    Parameters:
        num_samples (int): Total number of samples to generate (distributed across 3 classes).
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing temperature_c, humidity_pct, and weather_condition.
    """
    np.random.seed(random_seed)
    
    # Samples per class
    n_sunny = int(num_samples * 0.36)   # ~360 samples
    n_cloudy = int(num_samples * 0.32)  # ~320 samples
    n_rainy = num_samples - n_sunny - n_cloudy  # ~320 samples
    
    # 1. Sunny Class:
    # Characterized by warmer temperatures and lower/moderate humidity
    sunny_temp = np.random.normal(loc=30.0, scale=4.2, size=n_sunny)
    sunny_humidity = np.random.normal(loc=38.0, scale=9.0, size=n_sunny)
    sunny_labels = ["Sunny"] * n_sunny
    
    # 2. Cloudy Class:
    # Characterized by moderate temperatures and moderate/higher humidity
    cloudy_temp = np.random.normal(loc=22.5, scale=4.0, size=n_cloudy)
    cloudy_humidity = np.random.normal(loc=68.0, scale=8.5, size=n_cloudy)
    cloudy_labels = ["Cloudy"] * n_cloudy
    
    # 3. Rainy Class:
    # Characterized by cooler temperatures and high relative humidity
    rainy_temp = np.random.normal(loc=17.5, scale=3.8, size=n_rainy)
    rainy_humidity = np.random.normal(loc=86.5, scale=6.5, size=n_rainy)
    rainy_labels = ["Rainy"] * n_rainy
    
    # Combine all classes
    all_temp = np.concatenate([sunny_temp, cloudy_temp, rainy_temp])
    all_humidity = np.concatenate([sunny_humidity, cloudy_humidity, rainy_humidity])
    all_labels = np.concatenate([sunny_labels, cloudy_labels, rainy_labels])
    
    # Add a small amount of realistic boundary overlap (ambient noise)
    # This prevents the classes from being unnaturally 100% separable
    temp_noise = np.random.normal(0, 0.8, size=num_samples)
    humidity_noise = np.random.normal(0, 1.5, size=num_samples)
    
    all_temp = np.clip(all_temp + temp_noise, 5.0, 45.0)
    all_humidity = np.clip(all_humidity + humidity_noise, 15.0, 99.0)
    
    # Create DataFrame
    df = pd.DataFrame({
        "temperature_c": np.round(all_temp, 1),
        "humidity_pct": np.round(all_humidity, 1),
        "weather_condition": all_labels
    })
    
    # Shuffle the dataset so samples aren't ordered by class
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "weather_data.csv")
    
    print("Generating synthetic educational weather dataset...")
    df = generate_weather_dataset(num_samples=1000, random_seed=42)
    df.to_csv(csv_path, index=False)
    
    print(f"Dataset successfully saved to: {csv_path}")
    print(f"Total samples: {len(df)}")
    print("\nClass distribution:")
    print(df["weather_condition"].value_counts())
    print("\nFirst 5 rows:")
    print(df.head())
