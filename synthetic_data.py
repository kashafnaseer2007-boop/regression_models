# synthetic_data.py
import numpy as np
import pandas as pd

print("Generating synthetic dataset for regression models...")

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Create features with different distributions
X = pd.DataFrame({
    'Income': np.random.normal(50000, 15000, n_samples),    # Normal distribution
    'Age': np.random.randint(20, 70, n_samples),           # Uniform integers
    'Rooms': np.random.randint(1, 6, n_samples),           # Uniform integers  
    'Distance': np.random.uniform(1, 20, n_samples)        # Uniform floats
})

# Create target variable (house prices) with linear relationship
y = 20000 + 300*X['Income']/1000 + 5000*X['Rooms'] + np.random.normal(0, 10000, n_samples)

# Print results
print("✓ Dataset created successfully!")
print(f"📊 Shape: {X.shape}")
print(f"📈 Features: {list(X.columns)}")
print("\nFirst 3 rows:")
print(X.head(3))
print(f"\nTarget (Price) first 3 values: {y[:3]}")
