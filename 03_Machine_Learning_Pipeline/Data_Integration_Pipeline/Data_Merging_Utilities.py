# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# import dask.dataframe as dd

# # 1. Load Datasets Efficiently ------------------------------------------------
# # Use Dask to load datasets with an increased sample size for proper inference
# def load_dataset(file_path, sep="\t", sample=10_000_000):  # Increase sample size to 10 MB
#     return dd.read_csv(file_path, sep=sep, sample=sample)

# genera_counts = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\genera.counts (2).tsv")
# metadata = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\metadata (1).tsv")
# mtb = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\mtb.tsv")
# species = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\species (4).tsv")
# species_counts = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\species.counts (2).tsv")
# genera = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\genera (1).tsv")
# lda = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\LDA_bacteria.csv", sep="\t")

# # 2. Merge Datasets -----------------------------------------------------------
# # Merge datasets using Dask for parallel processing
# microbiome = dd.merge(genera_counts, species_counts, on="Sample", how="inner")
# microbiome = dd.merge(microbiome, species, on="Sample", how="inner")
# microbiome = dd.merge(microbiome, genera, on="Sample", how="inner")

# merged = dd.merge(metadata, microbiome, on="Sample", how="inner")
# full_data = dd.merge(merged, mtb, on="Sample", how="inner")

# # Convert to Pandas DataFrame for further processing
# full_data = full_data.compute()

# # 3. Process LDA Data ---------------------------------------------------------
# lda[['taxon', 'score']] = lda['scientific name,LDA'].str.split(',', expand=True)
# lda['score'] = lda['score'].astype(float)
# taxon_weights = lda.set_index('taxon')['score'].to_dict()

# # Utility function to match LDA taxa to column names
# def find_taxon_columns(taxon_name, all_columns):
#     patterns = [
#         f';g__{taxon_name}',  # Genus match
#         f';s__{taxon_name}',  # Species match
#         f';s__{taxon_name} '  # Handle subspecies
#     ]
#     return [col for col in all_columns if any(p in col for p in patterns)]

# # Get all microbiome-related columns
# micro_cols = [col for col in full_data.columns if any(x in col for x in ['g__', 's__'])]

# # Batch creation of LDA-weighted features
# lda_features = {}
# for taxon, weight in taxon_weights.items():
#     matched_cols = find_taxon_columns(taxon, micro_cols)
#     if matched_cols:
#         # Calculate weighted sum and store in a dictionary
#         lda_features[f'LDA_{taxon}'] = full_data[matched_cols].sum(axis=1) * abs(weight)

# # Add all LDA features to the DataFrame at once using pd.concat()
# lda_df = pd.DataFrame(lda_features)
# full_data = pd.concat([full_data, lda_df], axis=1)

# # 4. Perform KNN Imputation ---------------------------------------------------
# # Select numeric columns for imputation
# numeric_cols = full_data.select_dtypes(include=np.number).columns.tolist()

# # Initialize KNNImputer instance and perform imputation
# imputer = KNNImputer(n_neighbors=5)
# imputed_values = imputer.fit_transform(full_data[numeric_cols])

# # Convert imputed values back to DataFrame with correct column names and index alignment
# imputed_df = pd.DataFrame(imputed_values, columns=numeric_cols, index=full_data.index)

# # Replace numeric columns in original DataFrame with imputed values
# full_data[numeric_cols] = imputed_df

# # 5. Save Final Dataset -------------------------------------------------------
# full_data.to_csv(r"C:\RESEARCH-PROJECT\IHMP\imputed_microbiome_dataset.csv", index=False)

# print(f"Final dataset shape: {full_data.shape}")
# print("\nColumns breakdown:")
# print(f"- Clinical/Meta: {len(metadata.columns)}")
# print(f"- Microbiome Counts: {len(genera_counts.columns) + len(species_counts.columns)}")
# print(f"- Microbiome Normalized: {len(species.columns) + len(genera.columns)}")
# print(f"- Metabolites: {len(mtb.columns)-1}")
# print(f"- LDA Features: {len(lda_df.columns)}")
import os
import pandas as pd
import torch
import numpy as np

# Define chunk size (adjust based on your system's memory)
CHUNK_SIZE = 100  # Process 100 rows at a time

# Step 1: Load and merge datasets in chunks
def load_and_merge_chunks(file_paths, chunk_size=CHUNK_SIZE):
    merged_chunks = []
    
    # Iterate over chunks of the first dataset (genera_counts)
    for chunk in pd.read_csv(file_paths['genera_counts'], sep='\t', chunksize=chunk_size):
        merged_chunk = chunk
        for key, path in file_paths.items():
            if key != 'genera_counts' and key != 'lda':  # Exclude LDA for now
                other_df = pd.read_csv(path, sep='\t')
                merged_chunk = merged_chunk.merge(other_df, on="Sample", how="outer")
        merged_chunks.append(merged_chunk)
    
    # Concatenate all chunks
    return pd.concat(merged_chunks, ignore_index=True)

# File paths
file_paths = {
    'genera_counts': r"C:\RESEARCH-PROJECT\IHMP\genera.counts (2).tsv",
    'metadata': r"C:\RESEARCH-PROJECT\IHMP\metadata (1).tsv",
    'mtb': r"C:\RESEARCH-PROJECT\IHMP\mtb.tsv",
    'species': r"C:\RESEARCH-PROJECT\IHMP\species (4).tsv",
    'species_counts': r"C:\RESEARCH-PROJECT\IHMP\species.counts (2).tsv",
    'genera': r"C:\RESEARCH-PROJECT\IHMP\genera (1).tsv",
    'lda': r"C:\RESEARCH-PROJECT\IHMP\LDA_bacteria.csv"
}

# Load and merge datasets
merged_data = load_and_merge_chunks(file_paths)

# Load LDA data separately
lda_data = pd.read_csv(file_paths['lda'])

# Step 2: Handle missing values with PyTorch
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
numeric_data = merged_data[numeric_columns].values

# Custom KNN imputation function with PyTorch
def torch_knn_imputation(data, k=5, chunk_size=CHUNK_SIZE):
    num_rows = data.shape[0]
    imputed_data = []
    
    # Convert full data to tensor once
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = data[start:end]
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
        
        # Replace NaNs with 0 temporarily
        mask = torch.isnan(chunk_tensor)
        data_filled = torch.where(mask, torch.tensor(0.0), chunk_tensor)
        
        # Compute distances in smaller batches
        distances = []
        for ref_start in range(0, num_rows, chunk_size):
            ref_end = min(ref_start + chunk_size, num_rows)
            ref_chunk = data_tensor[ref_start:ref_end]
            # Broadcast and compute squared differences
            diff = data_filled.unsqueeze(1) - ref_chunk.unsqueeze(0)
            dist = torch.sum(diff ** 2, dim=2)
            distances.append(dist)
        
        distances = torch.cat(distances, dim=1)
        
        # Mask self-distances
        eye = torch.eye(chunk_tensor.shape[0], data.shape[0], dtype=torch.bool)
        distances = torch.where(eye, torch.tensor(float('inf')), distances)
        
        # Get k-nearest neighbors
        _, indices = torch.topk(-distances, k=k, dim=1)
        
        # Gather neighbor values
        neighbor_values = data_tensor[indices]
        
        # Compute mean of neighbors
        imputed_values = torch.mean(neighbor_values, dim=1)
        
        # Replace NaN with imputed values
        imputed_chunk = torch.where(mask, imputed_values, chunk_tensor)
        imputed_data.append(imputed_chunk.numpy())
    
    return np.concatenate(imputed_data, axis=0)

# Perform imputation
imputed_data = torch_knn_imputation(numeric_data, k=5, chunk_size=CHUNK_SIZE)
merged_data[numeric_columns] = imputed_data

# Step 3: Add LDA scores by matching bacterial names
def extract_taxonomic_name(column_name):
    parts = column_name.split(';')
    for part in parts[::-1]:
        if part.startswith('g__') or part.startswith('s__'):
            return part.split('__')[-1]
    return None

taxonomic_columns = [col for col in merged_data.columns if 'dBacteria' in col]
lda_mapping = dict(zip(lda_data['scientific name'], lda_data['LDA']))

for col in taxonomic_columns:
    tax_name = extract_taxonomic_name(col)
    if tax_name and tax_name in lda_mapping:
        col_tensor = torch.tensor(merged_data[col].values, dtype=torch.float32)
        lda_weighted = col_tensor * lda_mapping[tax_name]
        merged_data[f"{col}_LDA"] = lda_weighted.numpy()

# Step 4: Save the combined dataset
merged_data.to_csv(r"C:\RESEARCH-PROJECT\IHMP\combined_microbiome_dataset_pytorch.csv", index=False)

# Print confirmation
print("Combined dataset shape:", merged_data.shape)
print("Combined dataset saved as 'combined_microbiome_dataset_pytorch.csv'")
print(merged_data.head())