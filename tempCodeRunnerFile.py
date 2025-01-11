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

