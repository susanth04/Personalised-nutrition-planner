import os
import pandas as pd
import tensorflow as tf
import numpy as np

# Suppress oneDNN custom operations warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure TensorFlow uses CPU only
tf.config.set_visible_devices([], 'GPU')  # Disable GPU, force CPU usage

# Define chunk size (adjust based on your system's memory)
CHUNK_SIZE = 100  # Process 100 rows at a time

# Step 1: Load and merge datasets in chunks
def load_and_merge_chunks(file_paths, chunk_size=CHUNK_SIZE):
    # Initialize an empty DataFrame for the merged result
    merged_chunks = []
    
    # Iterate over chunks of the first dataset (genera_counts) as the base
    for chunk in pd.read_csv(file_paths['genera_counts'], sep='\t', chunksize=chunk_size):
        # Merge with other datasets one at a time
        merged_chunk = chunk
        for key, path in file_paths.items():
            if key != 'genera_counts' and key != 'lda':  # Exclude LDA for now
                other_df = pd.read_csv(path, sep='\t')
                merged_chunk = merged_chunk.merge(other_df, on="Sample", how="outer")
        merged_chunks.append(merged_chunk)
    
    # Concatenate all chunks
    return pd.concat(merged_chunks, ignore_index=True)

# File paths dictionary corrected with exact paths
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

# Load LDA data separately (small enough to load fully)
lda_data = pd.read_csv(file_paths['lda'])

# Step 2: Handle missing values with TensorFlow in chunks
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns

# Custom KNN imputation function with chunking
def tf_knn_imputation_chunked(data, k=5, chunk_size=CHUNK_SIZE):
    imputed_data = []
    num_rows = data.shape[0]
    
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = data[start:end]
        chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)
        
        # Replace NaNs with 0 temporarily
        mask = tf.math.is_nan(chunk_tensor)
        data_filled = tf.where(mask, 0.0, chunk_tensor)
        
        # Compute distances to the full dataset in smaller batches
        distances = []
        for ref_start in range(0, num_rows, chunk_size):
            ref_end = min(ref_start + chunk_size, num_rows)
            ref_chunk = data[ref_start:ref_end]
            ref_tensor = tf.convert_to_tensor(ref_chunk, dtype=tf.float32)
            dist = tf.reduce_sum(tf.square(tf.expand_dims(data_filled, 1) - tf.expand_dims(ref_tensor, 0)), axis=2)
            distances.append(dist)
        
        distances = tf.concat(distances, axis=1)
        
        # Mask self-distances
        distances = tf.where(tf.eye(tf.shape(chunk_tensor)[0], num_columns=tf.shape(data)[0], dtype=tf.bool), tf.float32.max, distances)
        
        # Get k-nearest neighbors
        _, indices = tf.nn.top_k(-distances, k=k)
        
        # Gather neighbor values from the full dataset
        neighbor_values = tf.gather(tf.convert_to_tensor(data, dtype=tf.float32), indices)
        
        # Compute mean of neighbors
        imputed_values = tf.reduce_mean(neighbor_values, axis=1)
        
        # Replace NaN with imputed values
        imputed_chunk = tf.where(mask, imputed_values, chunk_tensor)
        imputed_data.append(imputed_chunk.numpy())
    
    return np.concatenate(imputed_data, axis=0)

# Perform imputation on numeric data
numeric_data = merged_data[numeric_columns].values
imputed_data = tf_knn_imputation_chunked(numeric_data, k=5, chunk_size=CHUNK_SIZE)
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
        col_tensor = tf.convert_to_tensor(merged_data[col].values, dtype=tf.float32)
        lda_weighted = col_tensor * lda_mapping[tax_name]
        merged_data[f"{col}_LDA"] = lda_weighted.numpy()

# Step 4: Save the combined dataset
merged_data.to_csv(r"C:\RESEARCH-PROJECT\IHMP\combined_microbiome_dataset_tf.csv", index=False)

# Print confirmation
print("Combined dataset shape:", merged_data.shape)
print("Combined dataset saved as 'combined_microbiome_dataset_tf.csv'")
print(merged_data.head())