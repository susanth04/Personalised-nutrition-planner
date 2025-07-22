import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# USDA API setup
USDA_API_KEY = "NKn1op3bsIOSjbEYWVGrGnkJhGc8IDW3gDd8u0Ok"
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
CACHE_FILE = "food_nutrients_cache.csv"

# Load or initialize nutrient cache
if os.path.exists(CACHE_FILE):
    nutrient_cache = pd.read_csv(CACHE_FILE)
else:
    nutrient_cache = pd.DataFrame(columns=["food_item", "calories", "fiber", "protein", "fat"])
    nutrient_cache.to_csv(CACHE_FILE, index=False)

# Load dataset
data = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\DONE\final_dataset_with_food.tsv", sep='\t')

# Fetch nutrients from USDA API
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_nutrients(food_name):
    # Check cache
    cached = nutrient_cache[nutrient_cache["food_item"] == food_name]
    if not cached.empty:
        return cached.iloc[0][["calories", "fiber", "protein", "fat"]].to_dict()
    
    try:
        params = {
            "query": food_name,
            "api_key": USDA_API_KEY,
            "pageSize": 1,
            "dataType": ["Foundation", "SR Legacy"]
        }
        response = requests.get(USDA_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("foods"):
            nutrients = {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}
            for nutrient in data["foods"][0]["foodNutrients"]:
                name = nutrient["nutrientName"]
                value = nutrient["value"]
                if name == "Energy" and nutrient["unitName"] == "KCAL":
                    nutrients["calories"] = value
                elif name == "Dietary Fiber":
                    nutrients["fiber"] = value
                elif name == "Protein":
                    nutrients["protein"] = value
                elif name == "Total lipid (fat)":
                    nutrients["fat"] = value
            # Update cache
            new_entry = pd.DataFrame({"food_item": [food_name], **nutrients})
            new_entry.to_csv(CACHE_FILE, mode='a', header=False, index=False)
            return nutrients
        return {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}
    except Exception as e:
        print(f"Error fetching nutrients for {food_name}: {e}")
        return {"calories": 0, "fiber": 0, "protein": 0, "fat": 0}

# Update calories and add nutrients
def update_nutrients(row):
    food = row["food_item"]
    if pd.isna(food):
        return pd.Series({"calories": 0, "fiber": 0, "protein": 0, "fat": 0})
    nutrients = get_nutrients(food)
    return pd.Series(nutrients)

# Apply nutrient mapping
nutrient_data = data.apply(update_nutrients, axis=1)
data["calories"] = nutrient_data["calories"]
data["fiber"] = nutrient_data["fiber"]
data["protein"] = nutrient_data["protein"]
data["fat"] = nutrient_data["fat"]

# Compute daily nutrient intake
data["date"] = pd.to_datetime(data["timestamp"]).dt.date
daily_nutrients = data.groupby(["Sample", "date"])[["calories", "fiber", "protein", "fat"]].sum().reset_index()
daily_nutrients = daily_nutrients.rename(columns={
    "calories": "daily_calories",
    "fiber": "daily_fiber",
    "protein": "daily_protein",
    "fat": "daily_fat"
})

# Merge daily nutrients
data = data.merge(daily_nutrients, on=["Sample", "date"], how="left", suffixes=("", "_daily"))

# Fill NaNs in new columns
for col in ["daily_calories", "daily_fiber", "daily_protein", "daily_fat"]:
    data[col] = data[col].fillna(0)

# Save updated dataset
data.to_csv("final_dataset_with_nutrients.tsv", sep='\t', index=False)
print("Updated dataset saved as 'final_dataset_with_nutrients.tsv'")
print("Data Head:\n", data.head())