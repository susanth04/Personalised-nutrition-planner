# ibd_gpu_preprocessing.py
import torch
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MicrobiomeProcessor:
    def __init__(self, device):
        self.device = device
        self.lda_taxa = None

    def load_data(self, paths):
        """Load datasets with proper CPU/GPU separation"""
        data = {}
        # Load metadata on CPU
        data['metadata'] = pd.read_csv(paths[1], sep="\t")
        
        # Load other data as GPU tensors
        tensor_paths = [paths[0], paths[2], paths[3], paths[4], paths[5]]
        tensor_names = ['genera_counts', 'mtb', 'species', 'species_counts', 'genera']
        for name, path in zip(tensor_names, tensor_paths):
            df = pd.read_csv(path, sep="\t").set_index('Sample')
            data[name] = torch.tensor(df.values, dtype=torch.float32, device=device)
        
        # LDA data
        data['lda'] = pd.read_csv(paths[6], sep="\t")
        return data

    def process_metadata(self, df):
        """CPU-based metadata processing"""
        # Copy to avoid modifying original
        df = df.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        
        # KNN imputation for numeric columns
        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Mode imputation for categorical columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Convert to tensor at the end
        return torch.tensor(df.values, dtype=torch.float32, device=self.device)

    def clr_transform(self, tensor):
        """Centered Log-Ratio Transform on GPU"""
        geom_mean = torch.exp(torch.mean(torch.log(tensor + 1e-9), dim=1))
        return torch.log(tensor + 1e-9) - torch.log(geom_mean).unsqueeze(1)

    def gpu_pca(self, tensor, n_components=30):
        """PyTorch SVD-based PCA"""
        U, S, V = torch.linalg.svd(tensor)
        components = V[:n_components]
        return tensor @ components.T

    def calculate_diversity(self, tensor):
        """Alpha diversity metrics on GPU"""
        shannon = -torch.sum(tensor * torch.log(tensor + 1e-9), dim=1)
        simpson = 1 - torch.sum(tensor**2, dim=1)
        return torch.stack([shannon, simpson], dim=1)

    def process_lda(self, lda_df):
        """Fix LDA score conversion and handle invalid data"""
        lda_features = lda_df.iloc[:,0].str.split(',', expand=True)
        lda_features.columns = ['taxon', 'score']
        
        # Convert score to numeric and handle errors
        lda_features['score'] = pd.to_numeric(
            lda_features['score'], 
            errors='coerce'
        )
        
        # Remove invalid entries and show warnings
        invalid_count = lda_features['score'].isna().sum()
        if invalid_count > 0:
            print(f"Warning: Dropped {invalid_count} invalid LDA entries")
            lda_features = lda_features.dropna(subset=['score'])
        
        self.lda_taxa = lda_features.nlargest(100, 'score')['taxon'].tolist()

    def process(self, data_paths, output_path):
        # Load all datasets
        data = self.load_data(data_paths)
        
        # Process LDA features with validation
        self.process_lda(data['lda'])
        
        # Process metadata with KNN imputation
        meta_processed = self.process_metadata(data['metadata'])
        
        # Process MTB (metabolites) with scaling and KNN imputation
        mtb_scaled = StandardScaler().fit_transform(data['mtb'].cpu())
        mtb_processed = torch.tensor(mtb_scaled, device=device)
        
        # Process microbiome datasets (CLR transformation + LDA feature selection)
        microbiome_data = {
            'genera': self.clr_transform(data['genera']),
            'species': self.clr_transform(data['species']),
            'genera_counts': self.clr_transform(data['genera_counts']),
            'species_counts': self.clr_transform(data['species_counts'])
        }
        
        for name in microbiome_data:
            cols_to_keep = [i for i in range(microbiome_data[name].shape[1]) 
                            if any(taxon in data[name].columns[i] for taxon in self.lda_taxa)]
            microbiome_data[name] = microbiome_data[name][:, cols_to_keep]
        
        # Dimensionality reduction using PCA
        pca_results = {}
        for name, tensor in microbiome_data.items():
            pca_results[name] = self.gpu_pca(tensor)
        
        # Calculate alpha diversity metrics
        diversity_metrics = {
            'genera': self.calculate_diversity(microbiome_data['genera']),
            'species': self.calculate_diversity(microbiome_data['species'])
        }
        
        # Combine all features into a single tensor
        combined_tensor = torch.cat([
            meta_processed,
            mtb_processed,
            *pca_results.values(),
            *diversity_metrics.values()
        ], dim=1)
        
        # Save processed tensor to file
        torch.save(combined_tensor, output_path)
    
if __name__ == "__main__":
    # Define paths to datasets (update these paths based on your setup)
    DATA_PATHS = [
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\genera.counts (2).tsv",
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\metadata (1).tsv",
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\mtb.tsv",
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\species (4).tsv",
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\species.counts (2).tsv",
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\genera (1).tsv",
      r"C:\RESEARCH-PROJECT\03_Machine_Learning_Pipeline\Data_Integration_Pipeline\LDA_bacteria.csv"
    ]
    
    OUTPUT_PATH = r"C:\RESEARCH-PROJECT\processed_ibd_gpu.pt"
    
    # Initialize processor and preprocess all datasets
    processor = MicrobiomeProcessor(device=device)
    processor.process(DATA_PATHS, OUTPUT_PATH)
