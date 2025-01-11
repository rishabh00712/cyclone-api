

# Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



cyclone2 = pd.read_csv("cyclone2.csv")
cyclone1 = pd.read_csv("cyclone1.csv")

cyclone_data = pd.concat([cyclone1, cyclone2], axis=1)
cyclone_data.head(10)

cols_to_drop = ['International Number ID', 'Tropical Cyclone Number',	"Chinese Tropical Cyclone Data","Tropical Cyclone End Record",	"Number of Hours between Paths per Line",	"Name",	'Time', 'Grade', 'Average Wind Speed']
cyclone1.drop(columns=cols_to_drop, inplace=True)

cyclone1

cyclone1.rename(columns={"Minimum Central Pressure": "Pressure"}, inplace=True)
cyclone1.rename(columns={"Maximum Wind Speed": "Wind_Speed"}, inplace=True)

cyclone1

cyclone2

cols_to_drop = ['Cyclone Number','Time','SiR34','SATSer']
cyclone2.drop(columns=cols_to_drop, inplace=True)
cyclone2.rename(columns={"Wind Speed":"Wind_Speed"}, inplace=True)

cyclone2

cyclone3 = pd.read_csv("cyclone3.csv")
cyclone3

cols_to_drop = ["index", 'FID',	'YEAR',	'MONTH', 'DAY',	'AD_TIME',	'BTID',	'NAME', 'CAT', 'BASIN',	'Shape_Leng']
cyclone3.drop(columns=cols_to_drop, inplace=True)

cyclone3

cyclone3.rename(columns={"LAT": "Latitude"}, inplace=True)
cyclone3.rename(columns={"LONG": "Longitude"}, inplace=True)
cyclone3.rename(columns={"WIND_KTS": "Wind_Speed"}, inplace=True)
cyclone3.rename(columns={"PRESSURE": "Pressure"}, inplace=True)
cyclone3

cyclone_dataset = pd.concat([cyclone1, cyclone2, cyclone3], axis=0)
cyclone_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd

# Assuming cyclone_dataset is already loaded

# Define features (X) and target (y)
X = cyclone_dataset[['Pressure', 'Wind_Speed']].values
y = cyclone_dataset['Wind_Speed'].values

# Train-test-validation split (70-15-15 split)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the enhanced MLP Model with two hidden layers and L2 regularization (weight decay)
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.gelu2(x)
        x = self.fc3(x)
        return x

# Hyperparameters
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 8  # A small number of neurons in the hidden layers
output_dim = 1  # Output is Wind_Speed prediction
learning_rate = 0.01
num_epochs = 1000
l2_lambda = 1  # L2 regularization strength

# Initialize the model
model = MLPModel(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
criterion = nn.MSELoss()  # Using MSE to predict Wind Speed
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)  # L2 regularization via weight_decay

# Training Loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train_tensor)

    # Compute loss
    loss = criterion(y_pred, y_train_tensor)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model and evaluate on the validation set
model.eval()
with torch.no_grad():
    # Predict on validation set
    y_pred_val = model(X_val_tensor)
    predicted_wind_speeds_val = y_pred_val.numpy().flatten()

    # Predict on test set
    y_pred_test = model(X_test_tensor)
    predicted_wind_speeds_test = y_pred_test.numpy().flatten()

# Calculate the average Wind Speed from the training set
average_wind_speed = np.mean(y_train)

# Helper function to classify as "Cyclone" or "No Cyclone" based on Wind Speed
def classify_cyclone(wind_speeds):
    return ['No Cyclone' if speed < average_wind_speed else 'Cyclone' for speed in wind_speeds]

# Convert predictions to "Cyclone" or "No Cyclone" based on the threshold
cyclical_predictions_val = classify_cyclone(predicted_wind_speeds_val)
cyclical_predictions_test = classify_cyclone(predicted_wind_speeds_test)

# Convert y_val and y_test to binary based on the same threshold used for predictions
y_val_binary = [1 if speed >= average_wind_speed else 0 for speed in y_val]
y_test_binary = [1 if speed >= average_wind_speed else 0 for speed in y_test]

# Calculate Accuracy and Precision for Validation Set
accuracy_val = accuracy_score(y_val_binary, [1 if pred == 'Cyclone' else 0 for pred in cyclical_predictions_val])
precision_val = precision_score(y_val_binary, [1 if pred == 'Cyclone' else 0 for pred in cyclical_predictions_val])

# Calculate Accuracy and Precision for Test Set
accuracy_test = accuracy_score(y_test_binary, [1 if pred == 'Cyclone' else 0 for pred in cyclical_predictions_test])
precision_test = precision_score(y_test_binary, [1 if pred == 'Cyclone' else 0 for pred in cyclical_predictions_test])

# Print the results
print(f"Validation Accuracy: {accuracy_val:.4f}, Validation Precision: {precision_val:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}, Test Precision: {precision_test:.4f}")

user_input = np.array([[980, 10]])  # Example: Pressure=980, Wind_Speed=90
# Standardize the user input using the same scaler
user_input_scaled = scaler.transform(user_input)

# Convert to PyTorch tensor
user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)

# Ensure the model is in evaluation mode
model.eval()
with torch.no_grad():
    # Predict wind speed
    user_prediction = model(user_input_tensor).numpy().flatten()

# Classify the prediction as 'Cyclone' or 'No Cyclone'
user_prediction_label = classify_cyclone(user_prediction)

# Print results
print(f"User Input: {user_input}")
print(f"Predicted Wind Speed: {user_prediction}")
print(f"Predicted Category: {user_prediction_label}")


# Save Model
torch.save(model.state_dict(), 'cyclone_detection_model.pth')
print("Model saved as 'cyclone_detection_model.pth'")

import pickle

# Save the scaler to a file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved as 'scaler.pkl'")
