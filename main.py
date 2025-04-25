import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load Data
try:
    df = pd.read_csv('voice.csv')
except FileNotFoundError:
    print("Error: voice.csv not found in the workspace.")
    exit()

# Check if 'label' column exists
if 'label' not in df.columns:
    print("Error: 'label' column not found in the dataset.")
    exit()

# 2. Feature Selection
# Convert label to numerical
le = LabelEncoder()
df['label_numeric'] = le.fit_transform(df['label'])

# Calculate correlations (only with numeric columns)
numeric_cols = df.select_dtypes(include=np.number)
if 'label_numeric' in numeric_cols.columns:
    correlations = numeric_cols.corr()['label_numeric'].abs().sort_values(ascending=False)
    # Exclude the label itself from the features
    top_features = correlations[1:6].index.tolist() # Get top 5 features excluding label_numeric
    print(f"Selected Features: {top_features}")
else:
    print("Error: No numeric columns found for correlation analysis after excluding the label.")
    exit()


# 3. Prepare Data
X = df[top_features]
y = df['label_numeric']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Build Model (Minimal complexity)
model = keras.Sequential(
    [
        layers.Input(shape=(len(top_features),)), # Input layer for 5 features
        layers.Dense(4, activation="relu", name="hidden_layer"),  # Small hidden layer with ReLU
        layers.Dense(1, activation="sigmoid", name="output_layer"), # Output layer for binary classification
    ]
)

model.summary() # Print model structure

# 5. Compile and Train
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("\nTraining model...")
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2, verbose=0) # Use verbose=0 for cleaner output
print("Training complete.")

# 6. Evaluate and Extract Weights
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get weights of the first layer (input -> hidden)
input_layer_weights = model.get_layer('hidden_layer').get_weights()[0]

# Print weights for each input feature connecting to the hidden layer neurons
print("\nInput Layer Weights (Features -> Hidden Neurons):")
for i, feature in enumerate(top_features):
    print(f"  Feature '{feature}': {input_layer_weights[i]}")

# Optional: Show weights connecting hidden layer to output neuron
# hidden_output_weights = model.get_layer('output_layer').get_weights()[0]
# print("\nHidden Layer -> Output Neuron Weights:")
# print(hidden_output_weights)