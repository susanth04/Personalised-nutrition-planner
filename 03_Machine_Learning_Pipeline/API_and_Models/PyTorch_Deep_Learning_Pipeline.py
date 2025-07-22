import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\DONE\integrated_multiomics_dataset.tsv", sep='\t')
print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")

# Check for zeros in nutrient columns
zero_calories = data[data['avg_daily_calories'] == 0]
if len(zero_calories) > 0:
    print(f"Warning: {len(zero_calories)} rows with avg_daily_calories = 0")
    # Optional: Replace zeros with mean (uncomment to apply)
    # data['avg_daily_calories'] = data['avg_daily_calories'].replace(0, data['avg_daily_calories'].mean())

# Define features and target
exclude_cols = ['Dataset', 'Sample', 'Subject', 'Study.Group', 'Gender', 'smoking status',
               'condition_numeric', 'normalized_inflammation', 
               'breakfast_day1', 'breakfast_day2', 'breakfast_day3',
               'lunch_day1', 'lunch_day2', 'lunch_day3',
               'dinner_day1', 'dinner_day2', 'dinner_day3']
features = [col for col in data.columns if col not in exclude_cols]
X = data[features]
y = data['normalized_inflammation']

# Split and scale, preserving indices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_indices = X_train.index
test_indices = X_test.index

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device).reshape(-1, 1)

# Define neural network
class NutritionNet(nn.Module):
    def __init__(self, input_size):
        super(NutritionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Initialize model, loss, optimizer
model = NutritionNet(input_size=X_train.shape[1]).to(device)
criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Early stopping
best_val_loss = float('inf')
patience = 10
counter = 0

# Training loop
epochs = 200
batch_size = 32
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / (len(X_train_tensor) / batch_size)
    
    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
    
    scheduler.step(val_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "nutrition_net_best.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load("nutrition_net_best.pth", weights_only=True))
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    mae = mean_absolute_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)
    print(f"Test MAE: {mae}")
    print(f"Test R²: {r2}")

# Save final model
torch.save(model.state_dict(), "nutrition_net_final.pth")
print("Model saved as 'nutrition_net_final.pth'")

# Save predictions
test_results = pd.DataFrame({
    'Sample': data.loc[test_indices, 'Sample'],
    'True_normalized_inflammation': y_test.values,
    'Predicted_normalized_inflammation': y_pred.flatten()
})
test_results.to_csv("pytorch_predictions.csv", index=False)
print("Predictions saved as 'pytorch_predictions.csv'")