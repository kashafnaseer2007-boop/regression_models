!pip install -q scikit-learn pandas numpy matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("✓ Basic imports successful!")

try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    print("✓ sklearn imports successful!")
except Exception as e:
    print(f"✗ Error with sklearn: {e}")
    print("="*60)
print("ATTEMPTING TO LOAD CALIFORNIA HOUSING DATASET...")
print("="*60)

try:
    california = fetch_california_housing()

    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = california.target
    
    print("✓ Dataset loaded successfully!")
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    print("\nFeature names:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nTarget (y) shape: {y.shape}")
    
    # Show first 2 rows
    print("\nFirst 2 rows of features:")
    print(X.head(2))
    
    print(f"\nFirst 5 target values: {y[:5]}")
    
except Exception as e:
    print(f"\n✗ Error loading dataset: {e}")
    print("\nTrying backup dataset...")
    
    # Backup: Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'Income': np.random.normal(50000, 15000, n_samples),
        'Age': np.random.randint(20, 70, n_samples),
        'Rooms': np.random.randint(1, 6, n_samples),
        'Distance': np.random.uniform(1, 20, n_samples)
    })
    y = 20000 + 0.3*X['Income'] + 5000*X['Rooms'] + np.random.normal(0, 10000, n_samples)
    
    print("✓ Created synthetic dataset as backup")
    print(f"Shape: {X.shape}")
    print("Features:", list(X.columns))

print("\n" + "="*60)
print("CELL 2 COMPLETE")
print("="*60)
print("="*60)
print("VISUALIZING OUR SYNTHETIC DATASET")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
# Without flatten: axes[0,0], axes[0,1], axes[1,0], axes[1,1] 
# With flatten: axes[0], axes[1], axes[2], axes[3]  ← easier to loop
# e.g;
# axes[0] # first graph (top-left)
# axes[1] # second graph (top-right)  
# axes[2] # third graph (bottom-left)
# axes[3] # fourth graph (bottom-right)
features = ['Income', 'Age', 'Rooms', 'Distance']
colors = ['blue', 'green', 'red', 'purple']
for i, feature in enumerate(features): # enumerate -> Just a shortcut to get index+value together.
    ax = axes[i]
    ax.scatter(X[feature], y, alpha=0.5, color=colors[i], s=20)
    ax.set_xlabel(feature)
    ax.set_ylabel('Price (units)')
    ax.set_title(f'{feature} vs Price')
    ax.grid(True, alpha=0.3)
    
    if feature in ['Income', 'Rooms']:
        z = np.polyfit(X[feature], y, 1)
# The 1 means: "fit a polynomial of degree 1" = straight line
# Degree examples:
# 1 = straight line (y = mx + b)
# 2 = parabola (y = ax² + bx + c)
# 3 = cubic curve
        p = np.poly1d(z)
        ax.plot(X[feature].sort_values(), p(X[feature].sort_values()), 
               'k--', linewidth=2, label='Trend')
plt.suptitle('How Each Feature Affects Price', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("\nWHAT YOU SHOULD SEE:")
print("✓ Income ↑ → Price ↑ (clear upward trend)")
print("✓ Rooms ↑ → Price ↑ (stepped pattern - rooms are whole numbers)")
print("✗ Age → No clear relationship (random scatter)")
print("✗ Distance → No clear relationship (random scatter)")
print(f"\nDATASET SUMMARY:")
print(f"• Samples: {X.shape[0]} houses")
print(f"• Features: {list(X.columns)}")
print(f"• Price range: {y.min():,.0f} to {y.max():,.0f} units")
print(f"• Average price: {y.mean():,.0f} units")
print("="*60)
print("SPLITTING AND SCALING DATA")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Data split successfully!")
print(f"Training set: {X_train.shape[0]} houses (80%)")
print(f"Test set: {X_test.shape[0]} houses (20%)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\n✓ Features scaled (mean=0, std=1)")
print("Why scale? Some models (like SVR, Ridge) work better with scaled data.")
print("Random Forest doesn't need scaling - we'll use unscaled data for it.")

print("\n" + "="*60)
print("DATA READY FOR MODELING!")
print("="*60)
print("\nSample from training set (first house):")
for i, col in enumerate(X.columns):
    print(f"  {col}: {X_train.iloc[0, i]:.2f} → Scaled: {X_train_scaled[0, i]:.2f}")
print(f"  Price: {y_train[0]:.2f}")
