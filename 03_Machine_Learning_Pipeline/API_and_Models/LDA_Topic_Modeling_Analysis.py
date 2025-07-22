import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Assuming df is your already-loaded and imputed dataset
df = pd.read_csv("IHMP\integrated_microbiome_dataset.tsv", sep='\t')

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# For genera data
genera_cols = [col for col in df.columns if col.startswith('genera_PC')]
X_genera = df[genera_cols]
y = df['condition_numeric']

scaler_genera = StandardScaler()
X_genera_scaled = scaler_genera.fit_transform(X_genera)

# Apply LDA
lda_genera = LinearDiscriminantAnalysis()
X_lda_genera = lda_genera.fit_transform(X_genera_scaled, y)

# Check shape and add components appropriately
if X_lda_genera.shape[1] == 1:
    df['genera_LDA'] = X_lda_genera
else:
    for i in range(X_lda_genera.shape[1]):
        df[f'genera_LDA{i+1}'] = X_lda_genera[:, i]

# For species data
species_cols = [col for col in df.columns if col.startswith('species_PC')]
X_species = df[species_cols]
scaler_species = StandardScaler()
X_species_scaled = scaler_species.fit_transform(X_species)
lda_species = LinearDiscriminantAnalysis()
X_lda_species = lda_species.fit_transform(X_species_scaled, y)

# Add species LDA components
if X_lda_species.shape[1] == 1:
    df['species_LDA'] = X_lda_species
else:
    for i in range(X_lda_species.shape[1]):
        df[f'species_LDA{i+1}'] = X_lda_species[:, i]

# For metabolite data
mtb_cols = [col for col in df.columns if col.startswith('mtb_PC')]
X_mtb = df[mtb_cols]
scaler_mtb = StandardScaler()
X_mtb_scaled = scaler_mtb.fit_transform(X_mtb)
lda_mtb = LinearDiscriminantAnalysis()
X_lda_mtb = lda_mtb.fit_transform(X_mtb_scaled, y)

# Add metabolite LDA components
if X_lda_mtb.shape[1] == 1:
    df['mtb_LDA'] = X_lda_mtb
else:
    for i in range(X_lda_mtb.shape[1]):
        df[f'mtb_LDA{i+1}'] = X_lda_mtb[:, i]

# Print number of components for verification
print(f"Number of genera LDA components: {X_lda_genera.shape[1]}")
print(f"Number of species LDA components: {X_lda_species.shape[1]}")
print(f"Number of mtb LDA components: {X_lda_mtb.shape[1]}")
# # 1. Generate LDA for genera data
# # Extract genera principal components
# genera_cols = [col for col in df.columns if col.startswith('genera_PC')]
# X_genera = df[genera_cols]
# y = df['condition_numeric']  # Using condition (CD vs UC) as the target class

# # Standardize features before LDA
# scaler_genera = StandardScaler()
# X_genera_scaled = scaler_genera.fit_transform(X_genera)

# # Apply LDA - n_components is min(n_classes-1, n_features)
# # Since you have 2 classes (CD=1, UC=2), you'll get 1 LDA component
# lda_genera = LinearDiscriminantAnalysis()
# X_lda_genera = lda_genera.fit_transform(X_genera_scaled, y)

# # Add genera LDA component to dataframe
# df['genera_LDA'] = X_lda_genera

# # 2. Generate LDA for species data
# species_cols = [col for col in df.columns if col.startswith('species_PC')]
# X_species = df[species_cols]

# # Standardize species features
# scaler_species = StandardScaler()
# X_species_scaled = scaler_species.fit_transform(X_species)

# # Apply LDA for species
# lda_species = LinearDiscriminantAnalysis()
# X_lda_species = lda_species.fit_transform(X_species_scaled, y)

# # Add species LDA component to dataframe
# df['species_LDA'] = X_lda_species

# # 3. Generate LDA for metabolite data
# mtb_cols = [col for col in df.columns if col.startswith('mtb_PC')]
# X_mtb = df[mtb_cols]

# # Standardize metabolite features
# scaler_mtb = StandardScaler()
# X_mtb_scaled = scaler_mtb.fit_transform(X_mtb)

# # Apply LDA for metabolites
# lda_mtb = LinearDiscriminantAnalysis()
# X_lda_mtb = lda_mtb.fit_transform(X_mtb_scaled, y)

# # Add metabolite LDA component to dataframe
# df['mtb_LDA'] = X_lda_mtb

# # Optional: Check LDA explained variance ratio
# print("Genera LDA explained variance ratio:", lda_genera.explained_variance_ratio_)
# print("Species LDA explained variance ratio:", lda_species.explained_variance_ratio_)
# print("Metabolite LDA explained variance ratio:", lda_mtb.explained_variance_ratio_)

# # You can also examine which features contribute most to the LDA components
# genera_coefficients = pd.DataFrame({
#     'Feature': genera_cols,
#     'Coefficient': lda_genera.coef_[0]
# })
# print("Top genera features for LDA discrimination:")
# print(genera_coefficients.sort_values('Coefficient', ascending=False).head(5))