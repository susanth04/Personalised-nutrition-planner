import pandas as pd
import requests
import numpy as np
import random
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# USDA API setup
USDA_API_KEY = os.getenv("USDA_API_KEY")
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
NUTRIENT_CACHE_FILE = "food_nutrients_cache.csv"

# Load or initialize nutrient cache
if os.path.exists(NUTRIENT_CACHE_FILE):
    nutrient_cache = pd.read_csv(NUTRIENT_CACHE_FILE)
    nutrient_cache = nutrient_cache.drop_duplicates(subset="food_item", keep="first")
    nutrient_cache.to_csv(NUTRIENT_CACHE_FILE, index=False)
    print(f"Loaded nutrient cache with {len(nutrient_cache)} unique food items")
else:
    nutrient_cache = pd.DataFrame(columns=["food_item", "calories", "fiber", "protein", "fat"])
    nutrient_cache.to_csv(NUTRIENT_CACHE_FILE, index=False)
    print("Created new nutrient cache file")

# Create dictionary for fast lookups
nutrient_cache_dict = {
    row["food_item"]: {
        "calories": row["calories"],
        "fiber": row["fiber"],
        "protein": row["protein"],
        "fat": row["fat"]
    } for _, row in nutrient_cache.iterrows()
}

# PyTorch imputation class
class ImputationAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super(ImputationAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to perform PyTorch imputation
def impute_with_pytorch(dataframe, id_columns):
    print("Starting PyTorch imputation...")
    imputed_df = dataframe.copy()
    numerical_columns = [col for col in dataframe.columns if col not in id_columns and pd.api.types.is_numeric_dtype(dataframe[col])]
    categorical_columns = [col for col in dataframe.columns if col not in id_columns and col not in numerical_columns]
    
    print(f"Found {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns")
    
    if categorical_columns:
        for col in categorical_columns:
            mode_value = dataframe[col].mode()[0]
            imputed_df[col] = dataframe[col].fillna(mode_value)
        print(f"Imputed {len(categorical_columns)} categorical columns using most frequent value")
    
    if not numerical_columns:
        print("No numerical columns found for PyTorch imputation")
        return imputed_df
    
    numerical_data = dataframe[numerical_columns].copy()
    mask = numerical_data.isna().to_numpy()
    for col in numerical_columns:
        col_mean = numerical_data[col].mean()
        numerical_data[col] = numerical_data[col].fillna(col_mean)
    
    means = numerical_data.mean()
    stds = numerical_data.std()
    normalized_data = (numerical_data - means) / stds
    data_tensor = torch.tensor(normalized_data.values, dtype=torch.float32)
    mask_tensor = torch.tensor(~mask, dtype=torch.float32)
    
    input_dim = len(numerical_columns)
    model = ImputationAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = min(64, len(data_tensor))
    epochs = 100
    dataset = TensorDataset(data_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Training autoencoder for imputation...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, batch_mask in dataloader:
            output = model(batch_data)
            loss = criterion(output * batch_mask, batch_data * batch_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    model.eval()
    with torch.no_grad():
        imputed_tensor = model(data_tensor)
    
    imputed_numpy = imputed_tensor.numpy() * stds.values + means.values
    imputed_numerical = pd.DataFrame(imputed_numpy, columns=numerical_columns, index=numerical_data.index)
    
    for col in numerical_columns:
        missing_idx = dataframe[col].isna()
        imputed_df.loc[missing_idx, col] = imputed_numerical.loc[missing_idx, col]
    
    print("PyTorch imputation completed")
    return imputed_df

# Load raw dataset
try:
    raw_dataset = pd.read_csv("C:/RESEARCH-PROJECT/IHMP/DONE/imputed_microbiome_dataset.tsv", sep='\t')
    print(f"Successfully loaded dataset with {raw_dataset.shape[0]} rows and {raw_dataset.shape[1]} columns")
    
    id_columns = ["Sample", "Subject"]
    SKIP_IMPUTATION = True
    if SKIP_IMPUTATION:
        final_dataset = raw_dataset
        print("Skipped PyTorch imputation (dataset is pre-imputed)")
    else:
        final_dataset = impute_with_pytorch(raw_dataset, id_columns)
        print("Dataset imputation complete")
except Exception as e:
    print(f"Error loading or imputing dataset: {e}")
    final_dataset = pd.DataFrame({
        "Sample": ["S1", "S2", "S3"],
        "Subject": ["P1", "P2", "P3"]
    })
    print("Using sample dataset for testing")

# Debug unique Samples
unique_samples = final_dataset["Sample"].unique()
print(f"Found {len(unique_samples)} unique Samples in final_dataset")

# Simulate food diary
ibd_foods = [
    "chicken breast", "turkey breast", "salmon", "tuna", "cod", "shrimp", "eggs", "tofu", "tempeh",
    "lean beef", "pork tenderloin", "white fish", "scallops", "chicken thigh", "lamb",
    "brown rice", "quinoa", "oatmeal", "buckwheat", "rice noodles", "gluten-free bread", "millet",
    "sorghum", "cornmeal", "gluten-free pasta", "wild rice", "polenta",
    "spinach", "kale", "carrots", "zucchini", "cucumber", "bell peppers", "eggplant", "green beans",
    "lettuce", "pumpkin", "bok choy", "bamboo shoots", "parsnips", "potatoes", "radish", "squash",
    "chard", "collard greens", "fennel", "celery",
    "banana", "blueberries", "strawberries", "raspberries", "kiwi", "pineapple", "cantaloupe",
    "oranges", "grapes", "papaya", "clementine", "watermelon", "dragon fruit",
    "almonds", "walnuts", "chia seeds", "flaxseeds", "pumpkin seeds", "sunflower seeds", "hemp seeds",
    "lactose-free yogurt", "almond milk", "coconut milk", "lactose-free cheese", "soy milk",
    "olive oil", "avocado", "ginger", "turmeric", "maple syrup", "rice cakes", "honey", "coconut oil"
]

print("Generating food diary data...")
food_diary_data = []
start_date = datetime(2025, 1, 1)
sample_count = len(unique_samples)
print(f"Generating food diary for {sample_count} unique samples")

for sample in unique_samples:
    subject_rows = final_dataset[final_dataset["Sample"] == sample]
    if len(subject_rows) == 0:
        print(f"Warning: No data found for sample {sample}")
        continue
    subject = subject_rows["Subject"].iloc[0]
    
    for day in range(3):
        for meal in ["breakfast", "lunch", "dinner"]:
            timestamp = start_date + timedelta(days=day, hours=random.randint(7, 19))
            food_item = random.choice(ibd_foods)
            food_diary_data.append({
                "Sample": sample,
                "Subject": subject,
                "timestamp": timestamp,
                "food_item": food_item,
                "meal": meal,
                "day": day + 1
            })

food_diary = pd.DataFrame(food_diary_data)
food_diary['timestamp'] = pd.to_datetime(food_diary['timestamp'])
print(f"Generated {len(food_diary)} food diary entries")

# Fetch nutrients from USDA API
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_nutrients(food_name):
    global nutrient_cache_dict
    
    if food_name in nutrient_cache_dict:
        return nutrient_cache_dict[food_name]
    
    try:
        params = {
            "query": food_name,
            "api_key": USDA_API_KEY,
            "pageSize": 1,
            "dataType": "Foundation,SR Legacy"
        }
        print(f"Requesting nutrients for '{food_name}'...")
        response = requests.get(USDA_API_URL, params=params, timeout=30)
        if response.status_code != 200:
            print(f"API error: Status code {response.status_code}")
            return {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}
        
        data = response.json()
        if not data.get("foods") or len(data["foods"]) == 0:
            print(f"No nutrient data found for '{food_name}'")
            return {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}
        
        nutrients = {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}
        for nutrient in data["foods"][0].get("foodNutrients", []):
            name = nutrient.get("nutrientName", "").lower()
            value = nutrient.get("value", 0)
            if name == "energy" and nutrient.get("unitName") == "KCAL":
                nutrients["calories"] = value
            elif "fiber" in name:
                nutrients["fiber"] = value
            elif name == "protein":
                nutrients["protein"] = value
            elif "fat" in name or "lipid" in name:
                nutrients["fat"] = value
        
        nutrient_cache_dict[food_name] = nutrients
        if food_name not in nutrient_cache_dict:
            with open(NUTRIENT_CACHE_FILE, 'a') as f:
                f.write(f"{food_name},{nutrients['calories']},{nutrients['fiber']},{nutrients['protein']},{nutrients['fat']}\n")
        
        print(f"Found nutrients for '{food_name}': {nutrients}")
        return nutrients
    except Exception as e:
        print(f"Error fetching nutrients for '{food_name}': {e}")
        return {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}

# Process unique food items
print("Processing unique food items...")
unique_foods = food_diary["food_item"].unique()
print(f"Found {len(unique_foods)} unique food items")

food_nutrients = {}
for i, food_item in enumerate(unique_foods):
    food_nutrients[food_item] = get_nutrients(food_item)
    if (i + 1) % 5 == 0 or i == len(unique_foods) - 1:
        print(f"Processed {i+1}/{len(unique_foods)} unique food items ({(i+1)/len(unique_foods)*100:.1f}%)")

# Add nutrients to food diary
print("Adding nutrients to food diary...")
food_diary["calories"] = food_diary["food_item"].map(lambda x: food_nutrients[x]["calories"])
food_diary["fiber"] = food_diary["food_item"].map(lambda x: food_nutrients[x]["fiber"])
food_diary["protein"] = food_diary["food_item"].map(lambda x: food_nutrients[x]["protein"])
food_diary["fat"] = food_diary["food_item"].map(lambda x: food_nutrients[x]["fat"])

# Compute daily nutrient intake
print("Computing daily nutrient intake...")
food_diary["date"] = food_diary["timestamp"].dt.date
daily_nutrients = food_diary.groupby(["Sample", "date"])[["calories", "fiber", "protein", "fat"]].sum().reset_index()
daily_nutrients = daily_nutrients.rename(columns={
    "calories": "daily_calories",
    "fiber": "daily_fiber",
    "protein": "daily_protein",
    "fat": "daily_fat"
})

# Aggregate nutrients per Sample (average over days)
print("Aggregating nutrients per Sample...")
aggregated_nutrients = daily_nutrients.groupby("Sample")[["daily_calories", "daily_fiber", "daily_protein", "daily_fat"]].mean().reset_index()
aggregated_nutrients = aggregated_nutrients.rename(columns={
    "daily_calories": "avg_daily_calories",
    "daily_fiber": "avg_daily_fiber",
    "daily_protein": "avg_daily_protein",
    "daily_fat": "avg_daily_fat"
})

# Pivot food items to separate columns (breakfast_day1, lunch_day1, ...)
print("Pivoting food items to meal columns...")
food_diary["meal_day"] = food_diary["meal"] + "_day" + food_diary["day"].astype(str)
food_items_pivot = food_diary.pivot_table(
    index="Sample",
    columns="meal_day",
    values="food_item",
    aggfunc="first"
).reset_index()

# Fill NaNs in pivot table
food_items_pivot = food_items_pivot.fillna("None")

# Merge aggregated nutrients and food items
aggregated_data = aggregated_nutrients.merge(food_items_pivot, on="Sample", how="left")

# Standardize Sample column
final_dataset['Sample'] = final_dataset['Sample'].astype(str).str.strip().str.upper()
aggregated_data['Sample'] = aggregated_data['Sample'].astype(str).str.strip().str.upper()

# Merge datasets
print("Merging datasets...")
merge_cols = ["Sample", "avg_daily_calories", "avg_daily_fiber", "avg_daily_protein", "avg_daily_fat"] + \
             [col for col in food_items_pivot.columns if col != "Sample"]
merged_data = final_dataset.merge(aggregated_data[merge_cols], on="Sample", how="left")

# Fill NaNs in new columns
for col in ["avg_daily_calories", "avg_daily_fiber", "avg_daily_protein", "avg_daily_fat"]:
    merged_data[col] = merged_data[col].fillna(0)
for col in [col for col in food_items_pivot.columns if col != "Sample"]:
    merged_data[col] = merged_data[col].fillna("None")

# Debug final row count
print(f"Final dataset has {merged_data.shape[0]} rows and {merged_data.shape[1]} columns")
missing_samples = set(final_dataset["Sample"]) - set(merged_data["Sample"])
if missing_samples:
    print(f"Warning: {len(missing_samples)} Samples missing in merged_data: {missing_samples}")

# Save merged dataset
output_file = "final_dataset_with_food_nutrients.tsv"
merged_data.to_csv(output_file, sep='\t', index=False)
print(f"Merged dataset with aggregated nutrient data and meal columns saved as '{output_file}'")
print("\nSummary statistics of merged data:")
print(merged_data.describe())
print("\nMerged Data Head:")
print(merged_data.head())

# Save imputed dataset
imputed_file = "imputed_microbiome_dataset.tsv"
final_dataset.to_csv(imputed_file, sep='\t', index=False)
print(f"\nImputed microbiome dataset saved as '{imputed_file}'")