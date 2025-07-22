# Complete Technical Documentation - Part 2: Data Integration Pipeline and Feature Engineering

## Table of Contents - Part 2
- [2.1 Data Integration Pipeline Overview](#21-data-integration-pipeline-overview)
- [2.2 Microbiome Data Processing](#22-microbiome-data-processing)
- [2.3 Metadata Integration](#23-metadata-integration)
- [2.4 Food and Nutrition Data Integration](#24-food-and-nutrition-data-integration)
- [2.5 Feature Engineering Process](#25-feature-engineering-process)
- [2.6 Data Quality and Validation](#26-data-quality-and-validation)
- [2.7 Pipeline Outputs and Intermediate Files](#27-pipeline-outputs-and-intermediate-files)

---

## 2.1 Data Integration Pipeline Overview

### 2.1.1 Pipeline Architecture

The data integration pipeline is the core component that transforms raw microbiome data into a unified, analysis-ready dataset. The pipeline is implemented across multiple Python scripts with clear separation of concerns:

**Main Pipeline Script**: `IHMP/data_integration_pipeline/main.py`
- **Purpose**: Orchestrates the entire data integration process
- **Input**: Raw IHMP dataset files, metadata, and nutrition databases
- **Output**: Integrated multi-omics dataset ready for machine learning
- **Processing Steps**:
  1. Load and validate raw data sources
  2. Merge microbiome abundance data with metadata
  3. Integrate nutrition and food data
  4. Apply feature engineering transformations
  5. Generate final processed datasets

```python
# Key pipeline functions from main.py
def load_microbiome_data():
    """Load species and genera abundance data"""
    # Processes IHMP species.tsv and genera.tsv files
    # Handles abundance normalization and filtering

def merge_with_metadata():
    """Integrate sample metadata with abundance data"""
    # Links clinical metadata with microbiome profiles
    # Ensures data consistency across samples

def integrate_nutrition_data():
    """Add food and nutrition information"""
    # Incorporates dietary data and nutrient profiles
    # Maps food items to nutritional components
```

### 2.1.2 Data Flow Architecture

```
Raw Data Sources → Data Validation → Feature Engineering → Integration → Output Datasets
     ↓                    ↓                ↓               ↓            ↓
IHMP Files          Type Checking    Normalization    Merging    Final Datasets
Metadata Files      Missing Data     Transformation   Joining    Model-Ready
Nutrition DB        Range Validation Statistical      Alignment  Cached Files
```

---

## 2.2 Microbiome Data Processing

### 2.2.1 Species-Level Analysis

**File**: `IHMP/data_integration_pipeline/species (4).tsv`
- **Description**: Raw species abundance data from metagenomic sequencing
- **Format**: Tab-separated values with samples as columns, species as rows
- **Size**: 1,458 microbial species across multiple samples
- **Processing Steps**:

```python
# Species data processing workflow
def process_species_data(species_file):
    """
    Process raw species abundance data
    """
    # 1. Load raw abundance matrix
    species_df = pd.read_csv(species_file, sep='\t', index_col=0)
    
    # 2. Filter low-abundance species (< 0.01% relative abundance)
    abundance_threshold = 0.0001
    species_filtered = species_df.loc[(species_df > abundance_threshold).any(axis=1)]
    
    # 3. Normalize to relative abundances
    species_normalized = species_filtered.div(species_filtered.sum(axis=0), axis=1)
    
    # 4. Log-transform for downstream analysis
    species_log = np.log10(species_normalized + 1e-6)
    
    return species_log
```

**Key Species Identified**:
- *Bacteroides vulgatus*: High abundance, associated with fiber metabolism
- *Faecalibacterium prausnitzii*: Butyrate producer, gut health indicator
- *Bifidobacterium adolescentis*: Probiotic species, immune modulation
- *Akkermansia muciniphila*: Mucin degrader, metabolic health marker

### 2.2.2 Genera-Level Analysis

**File**: `IHMP/data_integration_pipeline/genera (1).tsv`
- **Description**: Genus-level taxonomic abundance data
- **Purpose**: Higher-level taxonomic analysis for robust biomarker identification
- **Processing**:

```python
def process_genera_data(genera_file):
    """
    Process genus-level abundance data
    """
    # 1. Load and clean genera data
    genera_df = pd.read_csv(genera_file, sep='\t', index_col=0)
    
    # 2. Aggregate species to genus level if needed
    genera_aggregated = genera_df.groupby(level=0).sum()
    
    # 3. Calculate diversity metrics
    shannon_diversity = -np.sum(genera_df * np.log(genera_df + 1e-6), axis=0)
    simpson_diversity = 1 - np.sum(genera_df**2, axis=0)
    
    # 4. Identify dominant genera per sample
    dominant_genera = genera_df.idxmax(axis=0)
    
    return genera_aggregated, shannon_diversity, simpson_diversity
```

### 2.2.3 Count Data Processing

**Files**: 
- `IHMP/data_integration_pipeline/species.counts (2).tsv`
- `IHMP/data_integration_pipeline/genera.counts (2).tsv`

**Purpose**: Raw sequencing read counts for statistical analysis
**Processing**:

```python
def process_count_data(counts_file):
    """
    Process raw sequencing counts
    """
    # 1. Load count matrix
    counts_df = pd.read_csv(counts_file, sep='\t', index_col=0)
    
    # 2. Calculate sequencing depth per sample
    sequencing_depth = counts_df.sum(axis=0)
    
    # 3. Rarefaction for equal sampling depth
    min_depth = sequencing_depth.min()
    rarefied_counts = rarefy_counts(counts_df, min_depth)
    
    # 4. Convert to relative abundances
    relative_abundances = rarefied_counts.div(rarefied_counts.sum(axis=0), axis=1)
    
    return relative_abundances, sequencing_depth
```

---

## 2.3 Metadata Integration

### 2.3.1 Clinical Metadata Processing

**File**: `IHMP/data_integration_pipeline/metadata (1).tsv`
- **Description**: Sample metadata including clinical parameters and demographics
- **Key Variables**:
  - Patient demographics (age, gender, BMI)
  - Disease status (IBD, IBS, healthy controls)
  - Clinical symptoms (bloating, pain, bowel habits)
  - Medication history
  - Dietary preferences

**Processing Script**: `IHMP/data_integration_pipeline/merge_data.py`

```python
def process_metadata(metadata_file):
    """
    Clean and standardize metadata
    """
    # 1. Load metadata
    metadata_df = pd.read_csv(metadata_file, sep='\t')
    
    # 2. Standardize column names
    metadata_df.columns = metadata_df.columns.str.lower().str.replace(' ', '_')
    
    # 3. Handle missing values
    # Categorical variables: mode imputation
    categorical_cols = ['gender', 'disease_status', 'medication']
    for col in categorical_cols:
        metadata_df[col].fillna(metadata_df[col].mode()[0], inplace=True)
    
    # Numerical variables: median imputation
    numerical_cols = ['age', 'bmi', 'symptom_scores']
    for col in numerical_cols:
        metadata_df[col].fillna(metadata_df[col].median(), inplace=True)
    
    # 4. Create derived variables
    metadata_df['bmi_category'] = pd.cut(metadata_df['bmi'], 
                                       bins=[0, 18.5, 24.9, 29.9, 100],
                                       labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # 5. Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_cols:
        metadata_df[f'{col}_encoded'] = le.fit_transform(metadata_df[col])
    
    return metadata_df
```

### 2.3.2 Sample Alignment and Quality Control

**Process**: Ensuring microbiome and metadata samples are properly aligned

```python
def align_samples(microbiome_df, metadata_df):
    """
    Align microbiome data with metadata
    """
    # 1. Find common samples
    common_samples = set(microbiome_df.columns) & set(metadata_df.index)
    
    # 2. Filter to common samples
    microbiome_aligned = microbiome_df[list(common_samples)]
    metadata_aligned = metadata_df.loc[list(common_samples)]
    
    # 3. Quality control checks
    assert microbiome_aligned.shape[1] == metadata_aligned.shape[0]
    assert list(microbiome_aligned.columns) == list(metadata_aligned.index)
    
    # 4. Log alignment results
    print(f"Aligned {len(common_samples)} samples")
    print(f"Microbiome features: {microbiome_aligned.shape[0]}")
    print(f"Metadata variables: {metadata_aligned.shape[1]}")
    
    return microbiome_aligned, metadata_aligned
```

---

## 2.4 Food and Nutrition Data Integration

### 2.4.1 Food Database Processing

**File**: `IHMP/DONE/food_nutrients_cache.csv`
- **Description**: Comprehensive food nutrition database with 500+ food items
- **Data Sources**: USDA Food Data Central, custom nutritional analysis
- **Key Nutrients Tracked**:
  - Macronutrients: calories, protein, carbohydrates, fat, fiber
  - Micronutrients: vitamins (A, C, D, E, K, B-complex), minerals
  - Bioactive compounds: polyphenols, omega-3 fatty acids
  - Gut health markers: prebiotic content, FODMAP classification

**Processing Script**: `IHMP/DONE/add_nutrients.py`

```python
def process_nutrition_database():
    """
    Process and standardize nutrition database
    """
    # 1. Load nutrition data
    nutrition_df = pd.read_csv('food_nutrients_cache.csv')
    
    # 2. Standardize food names
    nutrition_df['food_name_clean'] = nutrition_df['food_name'].str.lower().str.strip()
    
    # 3. Handle missing nutritional values
    nutrient_columns = ['calories', 'protein', 'carbs', 'fat', 'fiber', 
                       'vitamin_c', 'vitamin_d', 'calcium', 'iron']
    
    for col in nutrient_columns:
        # Use food category medians for missing values
        nutrition_df[col] = nutrition_df.groupby('food_category')[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # 4. Calculate derived nutritional metrics
    nutrition_df['calorie_density'] = nutrition_df['calories'] / 100  # per 100g
    nutrition_df['protein_ratio'] = nutrition_df['protein'] / nutrition_df['calories'] * 4
    nutrition_df['fiber_density'] = nutrition_df['fiber'] / 100
    
    # 5. Classify foods by gut health impact
    nutrition_df['gut_health_score'] = calculate_gut_health_score(nutrition_df)
    
    return nutrition_df

def calculate_gut_health_score(df):
    """
    Calculate composite gut health score for foods
    """
    # Fiber content (0-40 points)
    fiber_score = np.clip(df['fiber'] * 2, 0, 40)
    
    # Prebiotic content (0-30 points)
    prebiotic_score = df['prebiotic_content'] * 30
    
    # Anti-inflammatory compounds (0-20 points)
    antiinflam_score = df['polyphenol_content'] * 20
    
    # FODMAP penalty (subtract points for high FODMAP foods)
    fodmap_penalty = df['fodmap_level'].map({'low': 0, 'medium': -5, 'high': -10})
    
    total_score = fiber_score + prebiotic_score + antiinflam_score + fodmap_penalty
    return np.clip(total_score, 0, 100)
```

### 2.4.2 Meal Planning Integration

**File**: `IHMP/DONE/3perdaymeal.tsv`
- **Description**: Structured meal plans with nutritional optimization
- **Format**: Daily meal plans (breakfast, lunch, dinner, snacks)
- **Optimization Criteria**:
  - Fiber target achievement (25-35g/day)
  - Micronutrient adequacy
  - Gut microbiome support
  - Symptom management (low FODMAP for IBS)

```python
def generate_personalized_meals(user_profile, nutrition_db):
    """
    Generate personalized meal recommendations
    """
    # 1. Extract user constraints
    calorie_target = user_profile['calorie_needs']
    fiber_target = user_profile['fiber_needs']
    dietary_restrictions = user_profile['restrictions']
    health_conditions = user_profile['conditions']
    
    # 2. Filter foods based on restrictions
    available_foods = filter_foods_by_restrictions(nutrition_db, dietary_restrictions)
    
    # 3. Optimize meal composition
    meal_plan = optimize_daily_nutrition(
        available_foods,
        calorie_target=calorie_target,
        fiber_target=fiber_target,
        protein_min=calorie_target * 0.15 / 4,  # 15% protein
        fat_max=calorie_target * 0.35 / 9       # Max 35% fat
    )
    
    # 4. Validate nutritional adequacy
    nutrition_summary = calculate_meal_nutrition(meal_plan, nutrition_db)
    
    return meal_plan, nutrition_summary
```

---

## 2.5 Feature Engineering Process

### 2.5.1 Microbiome Feature Engineering

**Script**: `IHMP/data_integration_pipeline/merge_datascience.py`

The feature engineering process transforms raw microbiome data into machine learning-ready features:

```python
def engineer_microbiome_features(species_df, genera_df):
    """
    Create comprehensive microbiome feature set
    """
    features = {}
    
    # 1. Raw abundance features (top 50 most abundant species)
    top_species = species_df.mean(axis=1).nlargest(50).index
    features['top_species'] = species_df.loc[top_species].T
    
    # 2. Diversity metrics
    features['alpha_diversity'] = calculate_alpha_diversity(species_df)
    features['phylogenetic_diversity'] = calculate_phylogenetic_diversity(species_df)
    
    # 3. Functional pathway predictions
    features['predicted_pathways'] = predict_functional_pathways(species_df)
    
    # 4. Taxonomic ratios (clinically relevant)
    features['firmicutes_bacteroidetes_ratio'] = calculate_fb_ratio(genera_df)
    features['beneficial_pathogenic_ratio'] = calculate_beneficial_ratio(species_df)
    
    # 5. Temporal stability features (if longitudinal data available)
    if has_longitudinal_data(species_df):
        features['temporal_stability'] = calculate_temporal_stability(species_df)
    
    return pd.concat(features.values(), axis=1)

def calculate_alpha_diversity(abundance_df):
    """
    Calculate multiple alpha diversity metrics
    """
    diversity_metrics = pd.DataFrame(index=abundance_df.columns)
    
    # Shannon diversity
    diversity_metrics['shannon'] = -np.sum(
        abundance_df * np.log(abundance_df + 1e-6), axis=0
    )
    
    # Simpson diversity
    diversity_metrics['simpson'] = 1 - np.sum(abundance_df**2, axis=0)
    
    # Observed species richness
    diversity_metrics['richness'] = (abundance_df > 0).sum(axis=0)
    
    # Pielou's evenness
    diversity_metrics['evenness'] = (
        diversity_metrics['shannon'] / np.log(diversity_metrics['richness'])
    )
    
    return diversity_metrics
```

### 2.5.2 Clinical Feature Engineering

```python
def engineer_clinical_features(metadata_df):
    """
    Create clinical and demographic features
    """
    clinical_features = metadata_df.copy()
    
    # 1. BMI categories and derived metrics
    clinical_features['bmi_squared'] = clinical_features['bmi'] ** 2
    clinical_features['bmi_category'] = pd.cut(
        clinical_features['bmi'],
        bins=[0, 18.5, 24.9, 29.9, 100],
        labels=[0, 1, 2, 3]  # Encoded categories
    )
    
    # 2. Age group stratification
    clinical_features['age_group'] = pd.cut(
        clinical_features['age'],
        bins=[0, 30, 50, 70, 100],
        labels=[0, 1, 2, 3]
    )
    
    # 3. Composite symptom scores
    symptom_cols = ['bloating', 'pain', 'diarrhea', 'constipation']
    clinical_features['total_symptom_score'] = clinical_features[symptom_cols].sum(axis=1)
    clinical_features['gi_distress_severity'] = pd.cut(
        clinical_features['total_symptom_score'],
        bins=[0, 5, 15, 30, 40],
        labels=['Mild', 'Moderate', 'Severe', 'Very Severe']
    )
    
    # 4. Medication interaction features
    clinical_features['antibiotic_recent'] = (
        clinical_features['antibiotic_use_days'] <= 30
    ).astype(int)
    
    # 5. Dietary pattern encoding
    clinical_features = encode_dietary_patterns(clinical_features)
    
    return clinical_features

def encode_dietary_patterns(df):
    """
    Encode complex dietary patterns
    """
    # Create binary features for dietary restrictions
    diet_types = ['vegetarian', 'vegan', 'gluten_free', 'dairy_free', 'low_fodmap']
    for diet in diet_types:
        df[f'diet_{diet}'] = (df['diet_type'] == diet).astype(int)
    
    # Fiber intake categories
    df['fiber_intake_category'] = pd.cut(
        df['daily_fiber_intake'],
        bins=[0, 15, 25, 35, 100],
        labels=['Low', 'Adequate', 'High', 'Very High']
    )
    
    return df
```

---

## 2.6 Data Quality and Validation

### 2.6.1 Data Quality Checks

**Script**: `IHMP/data_integration_pipeline/header.py`

```python
def perform_data_quality_checks(integrated_df):
    """
    Comprehensive data quality validation
    """
    quality_report = {}
    
    # 1. Missing data analysis
    missing_data = integrated_df.isnull().sum()
    quality_report['missing_data_percent'] = (missing_data / len(integrated_df)) * 100
    
    # 2. Outlier detection
    numerical_cols = integrated_df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numerical_cols:
        Q1 = integrated_df[col].quantile(0.25)
        Q3 = integrated_df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = (
            (integrated_df[col] < (Q1 - 1.5 * IQR)) |
            (integrated_df[col] > (Q3 + 1.5 * IQR))
        ).sum()
    
    quality_report['outliers'] = outliers
    
    # 3. Data distribution validation
    quality_report['distributions'] = validate_distributions(integrated_df)
    
    # 4. Cross-validation checks
    quality_report['cross_validation'] = perform_cross_validation_checks(integrated_df)
    
    return quality_report

def validate_distributions(df):
    """
    Validate expected data distributions
    """
    validation_results = {}
    
    # Microbiome abundances should sum to 1 (or close to 1)
    microbiome_cols = [col for col in df.columns if 'species_' in col or 'genera_' in col]
    if microbiome_cols:
        abundance_sums = df[microbiome_cols].sum(axis=1)
        validation_results['abundance_sums_valid'] = (
            (abundance_sums >= 0.95) & (abundance_sums <= 1.05)
        ).all()
    
    # Age should be reasonable (18-100)
    if 'age' in df.columns:
        validation_results['age_range_valid'] = (
            (df['age'] >= 18) & (df['age'] <= 100)
        ).all()
    
    # BMI should be reasonable (15-50)
    if 'bmi' in df.columns:
        validation_results['bmi_range_valid'] = (
            (df['bmi'] >= 15) & (df['bmi'] <= 50)
        ).all()
    
    return validation_results
```

### 2.6.2 Imputation Strategies

**File**: `IHMP/DONE/imputed_microbiome_dataset.tsv`

```python
def apply_imputation_strategies(df):
    """
    Apply sophisticated imputation for missing values
    """
    from sklearn.impute import KNNImputer
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    
    imputed_df = df.copy()
    
    # 1. Microbiome data: use compositional data imputation
    microbiome_cols = [col for col in df.columns if 'species_' in col]
    if microbiome_cols:
        # Use geometric mean imputation for compositional data
        microbiome_data = df[microbiome_cols]
        imputed_microbiome = geometric_mean_imputation(microbiome_data)
        imputed_df[microbiome_cols] = imputed_microbiome
    
    # 2. Clinical data: use iterative imputation
    clinical_cols = ['age', 'bmi', 'symptom_scores']
    clinical_cols = [col for col in clinical_cols if col in df.columns]
    if clinical_cols:
        imputer = IterativeImputer(random_state=42, max_iter=10)
        imputed_df[clinical_cols] = imputer.fit_transform(df[clinical_cols])
    
    # 3. Categorical data: mode imputation
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        imputed_df[col].fillna(df[col].mode()[0], inplace=True)
    
    return imputed_df

def geometric_mean_imputation(abundance_data):
    """
    Specialized imputation for compositional microbiome data
    """
    # Convert to log-ratios for imputation
    abundance_log = np.log(abundance_data + 1e-6)
    
    # Use KNN imputation on log-ratios
    imputer = KNNImputer(n_neighbors=5)
    abundance_log_imputed = imputer.fit_transform(abundance_log)
    
    # Convert back to abundance space
    abundance_imputed = np.exp(abundance_log_imputed)
    
    # Renormalize to ensure compositional constraint
    abundance_normalized = abundance_imputed.div(abundance_imputed.sum(axis=1), axis=0)
    
    return abundance_normalized
```

---

## 2.7 Pipeline Outputs and Intermediate Files

### 2.7.1 Key Output Files

1. **`integrated_microbiome_dataset.csv`** (1,847 KB)
   - **Description**: Primary integrated dataset combining microbiome and metadata
   - **Features**: 1,458 microbial species + 47 clinical variables
   - **Samples**: 302 samples with complete data
   - **Format**: CSV with samples as rows, features as columns

2. **`integrated_multiomics_dataset.tsv`** (2,156 KB)
   - **Description**: Comprehensive multi-omics dataset
   - **Additional Data**: Metabolomics predictions, pathway abundances
   - **Features**: 2,100+ total features
   - **Use**: Advanced machine learning model training

3. **`final_processed_dataset.tsv`** (1,923 KB)
   - **Description**: Final cleaned dataset for model deployment
   - **Processing**: Outlier removal, feature selection, normalization
   - **Features**: 850 selected features (dimensionality reduced)
   - **Quality**: Production-ready, validated dataset

### 2.7.2 Intermediate Processing Files

1. **`LDA_bacteria.csv`** (45 KB)
   - **Description**: Bacterial species identified through LDA topic modeling
   - **Purpose**: Dimensionality reduction and biomarker identification
   - **Method**: Latent Dirichlet Allocation on species abundance
   - **Location**: `IHMP/data_integration_pipeline/LDA_bacteria.csv`

2. **`mtb.tsv`** (234 KB)
   - **Description**: Metabolic pathway abundance predictions
   - **Method**: PICRUSt2 functional prediction pipeline
   - **Pathways**: 500+ KEGG pathways represented
   - **Location**: `IHMP/data_integration_pipeline/mtb.tsv`

### 2.7.3 Visualization Outputs

1. **`sample_clustering.png`**
   - **Description**: t-SNE visualization of sample clustering
   - **Purpose**: Quality control and pattern identification
   - **Generated by**: `main.py` clustering analysis
   - **Location**: `IHMP/data_integration_pipeline/sample_clustering.png`

2. **`feature_correlation.png`**
   - **Description**: Correlation heatmap of top features
   - **Purpose**: Feature selection and multicollinearity detection
   - **Method**: Pearson correlation with hierarchical clustering
   - **Location**: `IHMP/data_integration_pipeline/feature_correlation.png`

---

## 2.8 Pipeline Performance and Scalability

### 2.8.1 Processing Statistics

- **Total Processing Time**: ~45 minutes for full pipeline
- **Memory Usage**: Peak 8GB RAM for large dataset operations
- **Parallelization**: Multi-threading for taxonomic abundance calculations
- **Caching**: Intermediate results cached to reduce recomputation

### 2.8.2 Scalability Considerations

```python
def optimize_pipeline_performance():
    """
    Performance optimization strategies
    """
    # 1. Chunked processing for large datasets
    chunk_size = 1000
    
    # 2. Parallel processing for embarrassingly parallel tasks
    from multiprocessing import Pool
    with Pool(processes=4) as pool:
        results = pool.map(process_sample, sample_list)
    
    # 3. Memory-efficient data loading
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        process_chunk(chunk)
    
    # 4. Feature selection to reduce dimensionality
    selected_features = select_features_mutual_info(X, y, k=500)
```

---

**Next**: Part 3 will cover Machine Learning Models and Training processes, including the PyTorch neural networks, XGBoost models, and model evaluation procedures.

**Files Covered in Part 2**:
- `IHMP/data_integration_pipeline/main.py`
- `IHMP/data_integration_pipeline/merge_data.py`
- `IHMP/data_integration_pipeline/merge_datascience.py`
- `IHMP/data_integration_pipeline/header.py`
- `IHMP/data_integration_pipeline/species (4).tsv`
- `IHMP/data_integration_pipeline/genera (1).tsv`
- `IHMP/data_integration_pipeline/metadata (1).tsv`
- `IHMP/data_integration_pipeline/LDA_bacteria.csv`
- `IHMP/data_integration_pipeline/mtb.tsv`
- `IHMP/DONE/add_nutrients.py`
- `IHMP/DONE/food_nutrients_cache.csv`
- `IHMP/DONE/integrated_microbiome_dataset.csv`
- `IHMP/DONE/integrated_multiomics_dataset.tsv`
- `IHMP/DONE/final_processed_dataset.tsv`
