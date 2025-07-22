import pandas as pd

def remove_lda_columns(df):
    """Remove columns containing LDA scores from DataFrame"""
    lda_cols = df.columns[df.columns.str.contains('LDA', case=False)]
    if not lda_cols.empty:
        print(f"Removing {len(lda_cols)} LDA-related columns")
        return df.drop(lda_cols, axis=1)
    print("No LDA columns found")
    return df

# Usage with your dataset
current_dataset = pd.read_csv(r'C:\RESEARCH-PROJECT\06_Processed_Datasets\unified_microbiome_dataset.csv', sep=',')  # Load your dataset
cleaned_dataset = remove_lda_columns(current_dataset)

# Save modified dataset
cleaned_dataset.to_csv('dataset_without_lda.csv', index=False)
