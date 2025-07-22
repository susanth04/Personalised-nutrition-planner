import pandas as pd
import numpy as np
import torch
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MicrobiomeDataProcessor:
    def __init__(self, use_gpu=True):
        """
        Initialize the microbiome data processor
        
        Parameters:
        -----------
        use_gpu : bool
            Whether to use GPU for processing if available
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize data containers
        self.genera_counts = None
        self.metadata = None
        self.mtb = None
        self.species_counts = None
        self.species = None
        self.genera = None
        self.lda = None
        
        # Initialize preprocessors
        self.imputer = KNNImputer(n_neighbors=5)
        self.scaler = StandardScaler()
    
    def load_specific_files(self):
        """
        Load data from specific file paths
        """
        print("Loading datasets from specific paths...")
        
        try:
            # Load each dataset using the specific file paths
            self.genera_counts = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\genera.counts (2).tsv", sep="\t")
            print(f"Loaded genera_counts: {self.genera_counts.shape}")
            
            self.metadata = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\metadata (1).tsv", sep="\t")
            print(f"Loaded metadata: {self.metadata.shape}")
            
            self.mtb = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\mtb.tsv", sep="\t")
            print(f"Loaded mtb (metabolites): {self.mtb.shape}")
            
            self.species_counts = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\species.counts (2).tsv", sep="\t")
            print(f"Loaded species_counts: {self.species_counts.shape}")
            
            self.species = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\species (4).tsv", sep="\t")
            print(f"Loaded species: {self.species.shape}")
            
            self.genera = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\genera (1).tsv", sep="\t")
            print(f"Loaded genera: {self.genera.shape}")
            
            self.lda = pd.read_csv(r"C:\RESEARCH-PROJECT\IHMP\LDA_bacteria.csv", sep="\t")
            print(f"Loaded LDA scores: {self.lda.shape}")
            
            # Set appropriate index for each dataset if needed
            if 'Sample' in self.genera_counts.columns:
                self.genera_counts.set_index('Sample', inplace=True)
                
            if 'Sample' in self.metadata.columns:
                self.metadata.set_index('Sample', inplace=True)
                
            if 'Sample' in self.mtb.columns:
                self.mtb.set_index('Sample', inplace=True)
                
            if 'Sample' in self.species_counts.columns:
                self.species_counts.set_index('Sample', inplace=True)
                
            if 'Sample' in self.species.columns:
                self.species.set_index('Sample', inplace=True)
                
            if 'Sample' in self.genera.columns:
                self.genera.set_index('Sample', inplace=True)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
        return self
    
    def clean_data(self):
        """Clean and preprocess all datasets"""
        print("Cleaning datasets...")
        
        # Clean metadata
        if self.metadata is not None:
            # Convert categorical variables
            cat_cols = ['Study.Group', 'Gender', 'race', 'smoking status']
            for col in cat_cols:
                if col in self.metadata.columns:
                    self.metadata[col] = self.metadata[col].astype('category')
            
            # Handle numeric columns with potential missing values
            num_cols = ['BMI_at_baseline', 'Height_at_baseline', 'Weight_at_baseline', 'fecalcal']
            for col in num_cols:
                if col in self.metadata.columns:
                    self.metadata[col] = pd.to_numeric(self.metadata[col], errors='coerce')
        
        # Clean LDA scores
        if self.lda is not None:
            # Ensure scientific name is the index for easier lookup
            if 'scientific name' in self.lda.columns:
                self.lda.set_index('scientific name', inplace=True)
        
        return self
    
    def handle_missing_values(self):
        """Handle missing values in the datasets"""
        print("Handling missing values...")
        
        # Function to impute missing values using KNN imputation
        def impute_dataframe(df):
            if df is None or df.empty:
                return None
                
            # Save categorical columns to reattach after imputation
            cat_cols = {}
            for col in df.select_dtypes(include=['category']).columns:
                cat_cols[col] = df[col]
                df = df.drop(col, axis=1)
                
            # Save non-numeric columns
            non_numeric = {}
            for col in df.select_dtypes(exclude=['number']).columns:
                non_numeric[col] = df[col]
                df = df.drop(col, axis=1)
            
            # Impute only numeric columns
            if not df.empty:
                imputed_array = self.imputer.fit_transform(df)
                df = pd.DataFrame(imputed_array, index=df.index, columns=df.columns)
            
            # Reattach non-numeric columns
            for col, values in non_numeric.items():
                df[col] = values
                
            # Reattach categorical columns
            for col, values in cat_cols.items():
                df[col] = values
                
            return df
        
        # Impute missing values in each dataset
        if self.metadata is not None:
            self.metadata = impute_dataframe(self.metadata)
            
        if self.mtb is not None:
            # For large metabolite dataset, use GPU-accelerated imputation
            if self.mtb.isna().sum().sum() > 0:
                print("GPU-accelerated imputation for metabolites dataset...")
                # Only select numeric columns for imputation
                numeric_cols = self.mtb.select_dtypes(include=['number']).columns
                
                # Convert to torch tensor for GPU processing
                mtb_tensor = torch.tensor(
                    self.mtb[numeric_cols].fillna(0).values, 
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Simple mean imputation using PyTorch (as KNN is memory-intensive)
                # Replace zeros (which were NaNs) with column means
                col_means = torch.mean(mtb_tensor, dim=0)
                mask = (mtb_tensor == 0)
                
                # Use broadcasting to replace zeros with means
                indices = torch.nonzero(mask)
                for idx in indices:
                    row, col = idx
                    mtb_tensor[row, col] = col_means[col]
                
                # Convert back to numpy and update dataframe
                imputed_array = mtb_tensor.cpu().numpy()
                self.mtb[numeric_cols] = imputed_array
                
        return self
    
    def normalize_data(self):
        """Normalize the datasets for machine learning"""
        print("Normalizing datasets...")
        
        # Function to normalize dataframe on GPU
        def normalize_on_gpu(df, columns):
            if df is None or df.empty:
                return None
        
            if columns is None or len(columns) == 0:
                return df
        
            # Copy dataframe to avoid modifying original
            df_norm = df.copy()
    
            # Convert to tensor and move to GPU
            tensor_data = torch.tensor(
                df[columns].values,
                dtype=torch.float32, 
                device=self.device
                )
    
            # Normalize (zero mean, unit variance)
        mean = tensor_data.mean(dim=0)
        std = tensor_data.std(dim=0)
        std[std == 0] = 1.0  # Avoid division by zero
        
        normalized = (tensor_data - mean) / std
        
        # Move back to CPU and update dataframe
        df_norm[columns] = normalized.cpu().numpy()
        
        return df_norm
            
        # Normalize microbiome abundance data
        if self.genera is not None:
            numeric_cols = self.genera.select_dtypes(include=['number']).columns
            self.genera = normalize_on_gpu(self.genera, numeric_cols)
            
        if self.species is not None:
            numeric_cols = self.species.select_dtypes(include=['number']).columns
            self.species = normalize_on_gpu(self.species, numeric_cols)
            
        if self.mtb is not None:
            numeric_cols = self.mtb.select_dtypes(include=['number']).columns
            self.mtb = normalize_on_gpu(self.mtb, numeric_cols)
            
        return self
    
    def engineer_features(self):
        """Create new features from existing data"""
        print("Engineering features...")
        
        # Calculate alpha diversity (if species data available)
        if self.species is not None:
            # Shannon diversity index calculation
            numeric_cols = self.species.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Move calculation to GPU
                species_tensor = torch.tensor(
                    self.species[numeric_cols].values,
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Normalize to get probabilities
                row_sums = species_tensor.sum(dim=1, keepdim=True)
                # Avoid division by zero
                row_sums[row_sums == 0] = 1.0
                species_probs = species_tensor / row_sums
                
                # Calculate Shannon index: -sum(p_i * log(p_i))
                # Replace 0s with small value to avoid log(0)
                epsilon = 1e-10
                species_probs = torch.clamp(species_probs, min=epsilon)
                log_probs = torch.log(species_probs)
                shannon_div = -torch.sum(species_probs * log_probs, dim=1)
                
                # Add diversity metrics to metadata
                if self.metadata is not None:
                    diversity_df = pd.DataFrame(
                        shannon_div.cpu().numpy(),
                        index=self.species.index,
                        columns=['shannon_diversity']
                    )
                    
                    # Check index alignment
                    if set(diversity_df.index).issubset(set(self.metadata.index)):
                        # Direct update
                        self.metadata.loc[diversity_df.index, 'shannon_diversity'] = diversity_df['shannon_diversity']
                    else:
                        # Create new entry in metadata
                        print("Warning: Shannon diversity indices could not be directly added to metadata due to index mismatch.")
                        self.diversity_metrics = diversity_df
                
                # Calculate Simpson's diversity index: 1 - sum(p_i^2)
                simpson_div = 1 - torch.sum(species_probs * species_probs, dim=1)
                
                # Add Simpson diversity to metadata or diversity metrics
                simpson_df = pd.DataFrame(
                    simpson_div.cpu().numpy(),
                    index=self.species.index,
                    columns=['simpson_diversity']
                )
                
                if hasattr(self, 'diversity_metrics'):
                    self.diversity_metrics['simpson_diversity'] = simpson_df['simpson_diversity']
                elif self.metadata is not None and set(simpson_df.index).issubset(set(self.metadata.index)):
                    self.metadata.loc[simpson_df.index, 'simpson_diversity'] = simpson_df['simpson_diversity']
                else:
                    self.diversity_metrics = simpson_df
        
        # Create composite features from metabolites if available
        if self.mtb is not None:
            # Example: Sum related metabolites (e.g., all SCFAs)
            scfa_columns = [col for col in self.mtb.columns if any(x in col.lower() for x in ['butyrate', 'acetate', 'propionate'])]
            if len(scfa_columns) > 0:
                self.mtb['total_scfa'] = self.mtb[scfa_columns].sum(axis=1)
        
        return self
    
    def prepare_for_unified_dataset(self):
        """Prepare data for creating a unified dataset by reducing dimensions"""
        print("Preparing for unified dataset...")
        
        # For genera data - select top taxa by variance
        if self.genera is not None:
            # Calculate variance for each column
            variances = self.genera.var()
            # Select top 50 genera by variance
            top_genera_cols = variances.nlargest(50).index.tolist()
            self.top_genera = self.genera[top_genera_cols]
            print(f"Selected top {len(top_genera_cols)} genera features")
        
        # For species data - select top taxa by variance
        if self.species is not None:
            # Calculate variance for each column
            variances = self.species.var()
            # Select top 50 species by variance
            top_species_cols = variances.nlargest(50).index.tolist()
            self.top_species = self.species[top_species_cols]
            print(f"Selected top {len(top_species_cols)} species features")
        
        # For metabolites data - select top by variance
        if self.mtb is not None:
            # Calculate variance for each column
            mtb_numeric = self.mtb.select_dtypes(include=['number'])
            variances = mtb_numeric.var()
            # Select top 50 metabolites by variance
            top_mtb_cols = variances.nlargest(50).index.tolist()
            self.top_mtb = self.mtb[top_mtb_cols]
            print(f"Selected top {len(top_mtb_cols)} metabolite features")
            
        # Process LDA scores if available
        if self.lda is not None:
            # Keep LDA scores as they are (typically already few in number)
            self.top_lda = self.lda.copy()
            
        return self
        
    def create_unified_dataset(self):
        """
        Create a single unified dataset from all preprocessed data sources
        
        Returns:
        --------
        pd.DataFrame: A single dataframe containing all relevant features
        """
        print("Creating unified dataset...")
        
        # Ensure we have prepared reduced-dimension datasets
        if not hasattr(self, 'top_genera') and self.genera is not None:
            self.prepare_for_unified_dataset()
            
        # Start with metadata as the base
        if self.metadata is None:
            print("Warning: Metadata is required for merging. Creating empty DataFrame.")
            unified_df = pd.DataFrame()
        else:
            unified_df = self.metadata.copy()
            
        # Function to safely merge datasets by sample ID
        def safe_merge(base_df, df_to_merge, prefix):
            if df_to_merge is None or df_to_merge.empty:
                return base_df
            
            # Ensure index alignment
            df_to_merge = df_to_merge.copy()
            
            # Add prefix to columns to avoid name conflicts
            df_to_merge.columns = [f"{prefix}_{col}" for col in df_to_merge.columns]
            
            # If indices are different types or have different names, try to align
            if base_df.index.name != df_to_merge.index.name:
                # Try to find a common column or index
                if 'Sample' in base_df.columns and df_to_merge.index.name == 'Sample':
                    return pd.merge(
                        base_df,
                        df_to_merge.reset_index(),
                        on='Sample',
                        how='left'
                    )
                elif base_df.index.name == 'Sample' and 'Sample' in df_to_merge.columns:
                    return pd.merge(
                        base_df.reset_index(),
                        df_to_merge,
                        on='Sample',
                        how='left'
                    ).set_index('Sample')
                else:
                    # Last resort: just reset indices and use position-based alignment
                    # This assumes the rows are in the same order in both dataframes
                    print(f"Warning: Could not find common index for {prefix} data. Using position-based alignment.")
                    merged_df = pd.concat([base_df.reset_index(drop=True), 
                                          df_to_merge.reset_index(drop=True)], axis=1)
                    if 'Sample' in merged_df.columns:
                        return merged_df.set_index('Sample')
                    else:
                        return merged_df
            else:
                # Direct merge on index
                return pd.merge(
                    base_df,
                    df_to_merge,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                
        # Add top genera features
        if hasattr(self, 'top_genera') and self.top_genera is not None:
            unified_df = safe_merge(unified_df, self.top_genera, 'genera')
            
        # Add top species features
        if hasattr(self, 'top_species') and self.top_species is not None:
            unified_df = safe_merge(unified_df, self.top_species, 'species')
        
        # Add top metabolite features
        if hasattr(self, 'top_mtb') and self.top_mtb is not None:
            unified_df = safe_merge(unified_df, self.top_mtb, 'mtb')
            
        # Add diversity metrics if they exist
        if hasattr(self, 'diversity_metrics') and self.diversity_metrics is not None:
            unified_df = safe_merge(unified_df, self.diversity_metrics, 'div')
            
        # Add LDA scores - this requires mapping taxa names
        if hasattr(self, 'top_lda') and self.top_lda is not None:
            # Create a features dataframe from LDA scores
            lda_df = pd.DataFrame()
            
            # Extract and prepare LDA scores as features
            if not self.top_lda.empty:
                if 'LDA' in self.top_lda.columns:
                    lda_df['LDA_score'] = self.top_lda['LDA'].values
                    lda_df.index = self.top_lda.index
                
                    # For now, just add a few summary statistics about LDA scores
                    # In a real application, you'd map these to the relevant taxa in your unified dataset
                    lda_stats = pd.DataFrame({
                        'lda_min_score': [lda_df['LDA_score'].min()],
                        'lda_max_score': [lda_df['LDA_score'].max()],
                        'lda_mean_score': [lda_df['LDA_score'].mean()],
                        'lda_median_score': [lda_df['LDA_score'].median()]
                    }, index=[0])
                    
                    # Replicate these stats for all samples
                    lda_stats_replicated = pd.DataFrame(
                        np.tile(lda_stats.values, (len(unified_df), 1)),
                        columns=lda_stats.columns,
                        index=unified_df.index
                    )
                    
                    # Add to unified dataset
                    unified_df = pd.concat([unified_df, lda_stats_replicated], axis=1)
                
        # Handle missing values in the final dataset
        missing_count = unified_df.isna().sum().sum()
        if missing_count > 0:
            print(f"Handling {missing_count} missing values in unified dataset...")
            
            # Only impute numeric columns
            numeric_cols = unified_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Convert to tensor for GPU processing
                tensor_data = torch.tensor(
                    unified_df[numeric_cols].fillna(0).values,
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Use simple mean imputation
                col_means = torch.mean(tensor_data, dim=0)
                mask = (tensor_data == 0)  # Assuming 0s are the filled NaN values
                
                # Replace zeros with column means
                indices = torch.nonzero(mask)
                for idx in indices:
                    row, col = idx
                    tensor_data[row, col] = col_means[col]
                
                # Update dataframe with imputed values
                unified_df[numeric_cols] = tensor_data.cpu().numpy()
            
            # For categorical columns, fill with mode
            cat_cols = unified_df.select_dtypes(include=['category', 'object']).columns
            for col in cat_cols:
                if unified_df[col].isna().any():
                    unified_df[col] = unified_df[col].fillna(unified_df[col].mode()[0])
                    
        print(f"Created unified dataset with shape: {unified_df.shape}")
        return unified_df
    
    def process_and_save_unified(self, output_dir, filename="unified_microbiome_dataset.csv"):
        """
        Process all data and save as a single unified CSV file
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files
        filename : str
            Name of the unified dataset file
        """
        # Process all data steps
        self.clean_data()
        self.handle_missing_values()
        self.normalize_data()
        self.engineer_features()
        self.prepare_for_unified_dataset()
        
        # Create unified dataset
        unified_data = self.create_unified_dataset()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save unified dataset
        output_path = os.path.join(output_dir, filename)
        unified_data.to_csv(output_path)
        print(f"Saved unified dataset to {output_path}")
        
        return unified_data
    
    def convert_to_pytorch(self, unified_data=None, target_column=None):
        """
        Convert processed data to PyTorch tensors ready for ML models
        
        Parameters:
        -----------
        unified_data : pd.DataFrame, optional
            Unified dataset to convert, if None, creates one
        target_column : str
            Column name to use as target variable
            
        Returns:
        --------
        dict: Dictionary containing feature tensors and target tensor
        """
        print("Converting to PyTorch tensors...")
        
        # Get unified data if not provided
        if unified_data is None:
            unified_data = self.create_unified_dataset()
            
        if unified_data is None or unified_data.empty:
            print("Error: No data available to convert")
            return None
            
        # Separate features and target
        X = unified_data.copy()
        y = None
        
        if target_column and target_column in X.columns:
            y = X[target_column].values
            X = X.drop(target_column, axis=1)
        
        # Handle non-numeric columns with one-hot encoding
        X = pd.get_dummies(X, drop_first=True)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        
        result = {'features': X_tensor, 'feature_names': X.columns.tolist()}
        
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
            result['target'] = y_tensor
            
        return result

# Example usage
if __name__ == "__main__":
    # Initialize processor with GPU support
    processor = MicrobiomeDataProcessor(use_gpu=True)
    
    # Load data from specific file paths
    processor.load_specific_files()
    
    # Process and save as a unified dataset
    output_dir = r"C:\RESEARCH-PROJECT\PROCESSED"
    unified_data = processor.process_and_save_unified(output_dir)
    
    # Optionally convert to PyTorch tensors
    pytorch_data = processor.convert_to_pytorch(unified_data, target_column="fecalcal")
    
    print("Data preprocessing complete!")