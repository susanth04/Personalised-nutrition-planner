import pandas as pd
import numpy as np
import torch
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Check if CUDA is available for GPU computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to load dataset from file path
def load_dataset(file_path):
    return pd.read_csv(file_path, sep="\t")

# Load all datasets using your file paths
print("Loading datasets...")
genera_counts_df = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\genera.counts (2).tsv")
metadata_df = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\metadata (1).tsv")
mtb_df = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\mtb.tsv")
species_df = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\species (4).tsv")
species_counts_df = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\species.counts (2).tsv")
genera_df = load_dataset(r"C:\RESEARCH-PROJECT\IHMP\genera (1).tsv")

print("Datasets loaded successfully")
print(f"Genera counts shape: {genera_counts_df.shape}")
print(f"Metadata shape: {metadata_df.shape}")
print(f"MTB shape: {mtb_df.shape}")
print(f"Species shape: {species_df.shape}")
print(f"Species counts shape: {species_counts_df.shape}")
print(f"Genera shape: {genera_df.shape}")

# Step 1: Filter and align samples across datasets
print("Finding common samples...")
def get_common_samples(dataframes):
    common_samples = set(dataframes[0]['Sample'])
    for df in dataframes[1:]:
        common_samples = common_samples.intersection(set(df['Sample']))
    return common_samples

# Define dataframes to find common samples across
dataframes_to_align = [genera_counts_df, metadata_df, mtb_df, species_df, species_counts_df, genera_df]
common_samples = get_common_samples(dataframes_to_align)
print(f"Number of common samples across all datasets: {len(common_samples)}")

# Filter all datasets to include only common samples
for i, df in enumerate(dataframes_to_align):
    dataframes_to_align[i] = df[df['Sample'].isin(common_samples)].copy()

genera_counts_df, metadata_df, mtb_df, species_df, species_counts_df, genera_df = dataframes_to_align

# Step 2: Identify biologically relevant features
print("Identifying biologically relevant features...")

# 2.1 For microbiome data, identify most abundant and most variable genera/species
def select_relevant_microbiome_features(df, top_n=100, method='abundance'):
    """Select most relevant microbiome features based on abundance or variance"""
    # Remove Sample column for calculations
    feature_df = df.drop('Sample', axis=1)
    
    if method == 'abundance':
        # Calculate total abundance across all samples
        feature_sums = feature_df.sum()
        # Get top_n most abundant features
        top_features = feature_sums.nlargest(top_n).index.tolist()
    elif method == 'variance':
        # Calculate variance across samples
        feature_vars = feature_df.var()
        # Get top_n most variable features
        top_features = feature_vars.nlargest(top_n).index.tolist()
    else:
        raise ValueError("Method must be 'abundance' or 'variance'")
    
    # Return columns of interest (Sample + top features)
    return ['Sample'] + top_features

# Select top genera and species based on abundance and variance
top_genera_abundance = select_relevant_microbiome_features(genera_df, top_n=50, method='abundance')
top_genera_variance = select_relevant_microbiome_features(genera_df, top_n=50, method='variance')
top_species_abundance = select_relevant_microbiome_features(species_df, top_n=100, method='abundance')
top_species_variance = select_relevant_microbiome_features(species_df, top_n=100, method='variance')

# Get unique features (combine abundance and variance features while removing duplicates)
relevant_genera = list(set(top_genera_abundance + top_genera_variance))
relevant_species = list(set(top_species_abundance + top_species_variance))

print(f"Selected {len(relevant_genera)-1} relevant genera")
print(f"Selected {len(relevant_species)-1} relevant species")

# 2.2 For metabolites, select most variable ones that might be relevant to IBD/inflammation
def select_relevant_metabolites(df, top_n=200):
    """Select most relevant metabolites based on variance and known connections to gut health"""
    # Remove Sample column for calculations
    feature_df = df.drop('Sample', axis=1)
    
    # Calculate variance across samples
    feature_vars = feature_df.var()
    
    # Find columns related to SCFAs, bile acids, etc. that are relevant to gut health
    gut_related_cols = [col for col in feature_df.columns if 
                        any(term in col.lower() for term in 
                           ['butyrate', 'propionate', 'acetate', 'lactate', 'bile', 
                            'fatty', 'acid', 'scfa', 'amino', 'tryptophan', 'histamine'])]
    
    # Combine approach: Include gut-related columns + top variable metabolites
    high_var_cols = feature_vars.nlargest(top_n).index.tolist()
    relevant_metabolites = list(set(gut_related_cols + high_var_cols))
    
    return ['Sample'] + relevant_metabolites

# Select relevant metabolites
relevant_mtb = select_relevant_metabolites(mtb_df, top_n=200)
print(f"Selected {len(relevant_mtb)-1} relevant metabolites")

# Step 3: Calculate biological diversity metrics using PyTorch
print("Calculating diversity metrics...")

def diversity_metrics_torch(df):
    """Calculate multiple diversity metrics using PyTorch for performance"""
    # Prepare data - remove Sample column
    samples = df['Sample'].values
    features = df.drop('Sample', axis=1).values
    
    # Convert to tensor and move to GPU
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    
    # Calculate row sums
    row_sums = torch.sum(features_tensor, dim=1, keepdim=True)
    
    # Calculate proportions (handle zeros to avoid log(0))
    proportions = features_tensor / (row_sums + 1e-10)  # Add small epsilon to avoid division by zero
    # Replace NaNs and Infs with zeros
    proportions = torch.nan_to_num(proportions, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Shannon diversity: -sum(p * log(p))
    epsilon = 1e-10
    log_proportions = torch.log(proportions + epsilon)
    shannon = -torch.sum(proportions * log_proportions, dim=1)
    
    # Simpson diversity: 1 - sum(p^2)
    simpson = 1 - torch.sum(proportions * proportions, dim=1)
    
    # Richness: count of non-zero elements in each sample
    richness = torch.sum((features_tensor > 0).float(), dim=1)
    
    # Move back to CPU and convert to numpy
    shannon_np = shannon.cpu().numpy()
    simpson_np = simpson.cpu().numpy()
    richness_np = richness.cpu().numpy()
    
    # Create dataframe with diversity metrics
    diversity_df = pd.DataFrame({
        'Sample': samples,
        'shannon_diversity': shannon_np,
        'simpson_diversity': simpson_np,
        'richness': richness_np
    })
    
    return diversity_df

# Calculate diversity metrics for both genera and species
genera_diversity = diversity_metrics_torch(genera_df)
species_diversity = diversity_metrics_torch(species_df)

# Create a combined diversity metrics dataframe
diversity_metrics = pd.merge(genera_diversity, species_diversity, on='Sample', how='left', suffixes=(None, '_species'))
# Rename the species columns
diversity_metrics = diversity_metrics.rename(columns={
    'shannon_diversity_species': 'species_shannon_diversity',
    'simpson_diversity_species': 'species_simpson_diversity',
    'richness_species': 'species_richness'
})

print("Diversity metrics calculated")

# Step 4: Handle missing values in metadata and prepare for integration
print("Processing metadata...")
# Select important clinical variables from metadata
clinical_vars = ['Dataset', 'Sample', 'Subject', 'Study.Group', 'Gender', 'Age', 'fecalcal', 
                'BMI_at_baseline', 'Height_at_baseline', 'Weight_at_baseline', 'smoking status']

# Filter metadata to include only variables of interest
# (keeping only columns that exist in the dataset)
metadata_selected = metadata_df[[col for col in clinical_vars if col in metadata_df.columns]]

# Handle missing values in numeric columns
metadata_numeric = metadata_selected.select_dtypes(include=[np.number])
metadata_non_numeric = metadata_selected.select_dtypes(exclude=[np.number])

if not metadata_numeric.empty:
    # Impute missing numeric values
    imputer = KNNImputer(n_neighbors=5)
    metadata_numeric_imputed = pd.DataFrame(
        imputer.fit_transform(metadata_numeric),
        columns=metadata_numeric.columns,
        index=metadata_numeric.index
    )
    # Combine back with non-numeric columns
    metadata_processed = pd.concat([metadata_non_numeric, metadata_numeric_imputed], axis=1)
else:
    metadata_processed = metadata_non_numeric

# Step 5: Prepare the integrated dataset
print("Building the integrated dataset...")

# 5.1 Extract the selected features from each dataset
genera_selected = genera_df[relevant_genera].copy()
species_selected = species_df[relevant_species].copy()
mtb_selected = mtb_df[relevant_mtb].copy()

# 5.2 Normalize features for better modeling
def normalize_features(df):
    """Normalize non-Sample columns to 0-1 range"""
    result = df.copy()
    for column in df.columns:
        if column != 'Sample':
            min_val = df[column].min()
            max_val = df[column].max()
            if max_val > min_val:  # Avoid division by zero
                result[column] = (df[column] - min_val) / (max_val - min_val)
    return result

# Normalize selected features
genera_normalized = normalize_features(genera_selected)
species_normalized = normalize_features(species_selected)
mtb_normalized = normalize_features(mtb_selected)

# 5.3 Dimensionality reduction for each dataset using PCA with PyTorch
class TorchPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.device = device
        self.components = None
        self.explained_variance_ratio = None
        
    def fit_transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        X_centered = X_tensor - X_tensor.mean(dim=0)
        
        # SVD decomposition
        U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Get components and explained variance
        self.components = V[:self.n_components].T
        explained_variance = (S**2) / (X.shape[0] - 1)
        self.explained_variance_ratio = explained_variance / torch.sum(explained_variance)
        
        # Project data onto principal components
        X_pca = torch.matmul(X_centered, self.components)
        
        return X_pca.cpu().numpy()

def reduce_dimensions(df, prefix, n_components=20):
    """Apply PCA to reduce dimensions while preserving biological signal"""
    # Make a copy and set Sample as index temporarily
    temp_df = df.copy()
    samples = temp_df['Sample'].values
    features = temp_df.drop('Sample', axis=1).values
    
    # Replace NaNs with zeros
    features[np.isnan(features)] = 0
    
    # Apply PCA
    pca = TorchPCA(n_components=n_components)
    pca_result = pca.fit_transform(features)
    
    # Create new dataframe with PCA components
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'{prefix}_PC{i+1}' for i in range(n_components)]
    )
    pca_df.insert(0, 'Sample', samples)
    
    # Calculate explained variance
    explained_variance = torch.sum(pca.explained_variance_ratio[:n_components]).item() * 100
    print(f"{prefix} - Explained variance with {n_components} components: {explained_variance:.2f}%")
    
    return pca_df

# Apply dimensionality reduction
genera_pca = reduce_dimensions(genera_normalized, 'genera', n_components=20)
species_pca = reduce_dimensions(species_normalized, 'species', n_components=30)
mtb_pca = reduce_dimensions(mtb_normalized, 'mtb', n_components=50)

# Step 6: Merge all processed datasets
print("Merging all datasets...")

# Start with metadata
integrated_df = metadata_processed.copy()

# Add in diversity metrics
integrated_df = pd.merge(integrated_df, diversity_metrics, on='Sample', how='left')

# Add dimensionality-reduced features
integrated_df = pd.merge(integrated_df, genera_pca, on='Sample', how='left')
integrated_df = pd.merge(integrated_df, species_pca, on='Sample', how='left')
integrated_df = pd.merge(integrated_df, mtb_pca, on='Sample', how='left')

# Step 7: Add target variables for modeling
if 'Study.Group' in integrated_df.columns:
    # Map study groups to numerical values for modeling
    group_mapping = {'CD': 1, 'UC': 2, 'nonIBD': 0}
    integrated_df['condition_numeric'] = integrated_df['Study.Group'].map(lambda x: group_mapping.get(x, -1))

# If fecalcal is available as an inflammation marker
if 'fecalcal' in integrated_df.columns:
    # Normalize to 0-1 range
    fc_min = integrated_df['fecalcal'].min()
    fc_max = integrated_df['fecalcal'].max()
    if fc_max > fc_min:
        integrated_df['normalized_inflammation'] = (integrated_df['fecalcal'] - fc_min) / (fc_max - fc_min)
    else:
        integrated_df['normalized_inflammation'] = 0  # Handle case where all values are the same

# Compute ratio of key bacterial phyla
if any('Firmicutes' in col for col in genera_df.columns) and any('Bacteroidetes' in col for col in genera_df.columns):
    # Find Firmicutes and Bacteroidetes columns
    firmicutes_cols = [col for col in genera_df.columns if 'Firmicutes' in col]
    bacteroidetes_cols = [col for col in genera_df.columns if 'Bacteroidetes' in col]
    
    if firmicutes_cols and bacteroidetes_cols:
        # Calculate sum for each phylum
        firmicutes_sum = genera_df[firmicutes_cols].sum(axis=1)
        bacteroidetes_sum = genera_df[bacteroidetes_cols].sum(axis=1)
        
        # Calculate F/B ratio (handling zeros)
        fb_ratio = pd.DataFrame({
            'Sample': genera_df['Sample'],
            'firmicutes_bacteroidetes_ratio': firmicutes_sum / (bacteroidetes_sum + 1e-10)
        })
        
        # Add to integrated dataset
        integrated_df = pd.merge(integrated_df, fb_ratio, on='Sample', how='left')

# Step 8: Save the final integrated dataset
# THIS IS THE FIX: Adding proper quoting and escaping to handle complex text fields
output_path = r"C:\RESEARCH-PROJECT\IHMP\integrated_microbiome_dataset.csv"
integrated_df.to_csv(output_path, index=False, sep=',', quoting=1)
print(f"Integrated dataset saved to: {output_path}")

# Alternate fix option - use tab delimiter instead of comma
output_path_alt = r"C:\RESEARCH-PROJECT\IHMP\integrated_microbiome_dataset.tsv"
integrated_df.to_csv(output_path_alt, index=False, sep='\t')
print(f"Integrated dataset also saved as TSV to: {output_path_alt}")

# Dataset summary
print(f"Final dataset shape: {integrated_df.shape}")
print(f"Number of features: {integrated_df.shape[1] - 1}")  # Excluding Sample column
print(f"Number of samples: {integrated_df.shape[0]}")

# Optional: Generate visualizations of the dataset
try:
    # 1. PCA visualization to see sample clustering
    plt.figure(figsize=(10, 8))
    if 'condition_numeric' in integrated_df.columns:
        scatter = plt.scatter(integrated_df['genera_PC1'], integrated_df['genera_PC2'], 
                 c=integrated_df['condition_numeric'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Condition (0=nonIBD, 1=CD, 2=UC)')
    else:
        plt.scatter(integrated_df['genera_PC1'], integrated_df['genera_PC2'], alpha=0.7)
    
    plt.title('Sample Clustering based on Genera PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(r"C:\RESEARCH-PROJECT\IHMP\sample_clustering.png")
    
    # 2. Correlation heatmap of key features
    plt.figure(figsize=(12, 10))
    # Select diverse set of features for correlation analysis
    key_features = ['shannon_diversity', 'simpson_diversity', 'richness']
    if 'normalized_inflammation' in integrated_df.columns:
        key_features.append('normalized_inflammation')
    if 'firmicutes_bacteroidetes_ratio' in integrated_df.columns:
        key_features.append('firmicutes_bacteroidetes_ratio')
    
    # Add some PCA components
    key_features.extend([f'genera_PC{i}' for i in range(1, 6)])
    key_features.extend([f'species_PC{i}' for i in range(1, 6)])
    key_features.extend([f'mtb_PC{i}' for i in range(1, 6)])
    
    # Filter features that exist in the dataset
    available_features = [f for f in key_features if f in integrated_df.columns]
    
    correlation_matrix = integrated_df[available_features].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(r"C:\RESEARCH-PROJECT\IHMP\feature_correlation.png")
    
    # 3. Add a visualization for diversity metrics across conditions
    if 'Study.Group' in integrated_df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Study.Group', y='shannon_diversity', data=integrated_df)
        plt.title('Shannon Diversity Across Conditions')
        plt.savefig(r"C:\RESEARCH-PROJECT\IHMP\diversity_by_condition.png")
        
        # 4. Visualization of F/B ratio if available
        if 'firmicutes_bacteroidetes_ratio' in integrated_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Study.Group', y='firmicutes_bacteroidetes_ratio', data=integrated_df)
            plt.title('Firmicutes/Bacteroidetes Ratio Across Conditions')
            plt.savefig(r"C:\RESEARCH-PROJECT\IHMP\fb_ratio_by_condition.png")
    
    print("Visualizations saved successfully")
except Exception as e:
    print(f"Could not generate visualizations: {e}")

print("\nData integration complete!")