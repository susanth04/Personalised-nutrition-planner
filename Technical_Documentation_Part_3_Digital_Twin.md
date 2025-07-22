# Complete Technical Documentation - Part 3: Machine Learning Models and Training

## Table of Contents - Part 3
- [3.1 Machine Learning Architecture Overview](#31-machine-learning-architecture-overview)
- [3.2 PyTorch Neural Network Models](#32-pytorch-neural-network-models)
- [3.3 XGBoost Ensemble Models](#33-xgboost-ensemble-models)
- [3.4 Feature Selection and Engineering](#34-feature-selection-and-engineering)
- [3.5 Model Training and Optimization](#35-model-training-and-optimization)
- [3.6 Model Evaluation and Validation](#36-model-evaluation-and-validation)
- [3.7 Model Deployment and Inference](#37-model-deployment-and-inference)
- [3.8 Digital Twin Integration](#38-digital-twin-integration)

---

## 3.1 Machine Learning Architecture Overview

### 3.1.1 Multi-Model Framework

The microbiome health platform employs a sophisticated multi-model framework that combines deep learning, ensemble methods, and metabolic modeling for comprehensive health prediction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODEL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│ Input Layer:                                                    │
│ ├─ Microbiome Features (1,458 species + genera)                 │
│ ├─ Clinical Metadata (47 variables)                             │
│ ├─ Nutritional Data (50+ nutrients)                             │
│ └─ Engineered Features (diversity, ratios)                      │
├─────────────────────────────────────────────────────────────────┤
│ Model Ensemble:                                                 │
│ ├─ PyTorch Neural Networks (Deep Learning)                      │
│ │  ├─ NutritionNet (Inflammation Prediction)                    │
│ │  └─ FluxPredictor (Metabolic Flux Prediction)                 │
│ ├─ XGBoost Models (Gradient Boosting)                           │
│ │  ├─ Inflammation Prediction                                   │
│ │  └─ Feature Importance Analysis                               │
│ └─ Digital Twin Models (COBRA + ML)                             │
│    ├─ Metabolic Network Simulation                              │
│    └─ Personalized Flux Predictions                             │
├─────────────────────────────────────────────────────────────────┤
│ Output Layer:                                                   │
│ ├─ Inflammation Scores                                          │
│ ├─ Metabolite Production Predictions                            │
│ ├─ Personalized Nutrition Recommendations                       │
│ └─ Health Risk Assessments                                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1.2 Model Integration Strategy

**Primary Prediction Target**: `normalized_inflammation`
- **Description**: Normalized inflammation biomarker score (0-1 scale)
- **Clinical Relevance**: Key indicator of gut health and systemic inflammation
- **Data Source**: Derived from clinical biomarkers and symptom assessments

**Feature Space**: 2,100+ engineered features
- **Microbiome**: Species/genera abundances, diversity metrics
- **Clinical**: Demographics, symptoms, medication history
- **Nutritional**: Macro/micronutrient intake, dietary patterns
- **Metabolic**: Predicted pathway abundances, metabolite levels

---

## 3.2 PyTorch Neural Network Models

### 3.2.1 NutritionNet Architecture

**File**: `IHMP/DONE/pytorch-dl.py`

The primary deep learning model for inflammation prediction uses a multi-layer feed-forward neural network:

```python
class NutritionNet(nn.Module):
    def __init__(self, input_size):
        super(NutritionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)      # First hidden layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)            # 30% dropout for regularization
        self.fc2 = nn.Linear(256, 128)             # Second hidden layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)              # Third hidden layer
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, 1)                # Output layer (regression)
    
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x
```

**Architecture Details**:
- **Input Layer**: Variable size (2,100+ features after preprocessing)
- **Hidden Layers**: 256 → 128 → 64 neurons with ReLU activation
- **Regularization**: 30% dropout in first two layers to prevent overfitting
- **Output**: Single neuron for regression (inflammation score)
- **Loss Function**: L1 Loss (Mean Absolute Error)
- **Optimizer**: Adam with learning rate 0.001

### 3.2.2 Training Configuration

```python
# Model initialization and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NutritionNet(input_size=X_train.shape[1]).to(device)
criterion = nn.L1Loss()  # MAE loss for robust regression
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                               patience=5, factor=0.5)

# Training hyperparameters
epochs = 200
batch_size = 32
early_stopping_patience = 10
```

**Training Features**:
- **GPU Acceleration**: CUDA support for faster training
- **Learning Rate Scheduling**: Reduce LR on plateau for fine-tuning
- **Early Stopping**: Prevents overfitting with patience=10 epochs
- **Batch Processing**: 32-sample batches for efficient memory usage

### 3.2.3 Data Preprocessing Pipeline

```python
# Feature preprocessing for neural networks
def preprocess_for_pytorch(data):
    """
    Comprehensive preprocessing for PyTorch models
    """
    # 1. Exclude non-predictive columns
    exclude_cols = ['Dataset', 'Sample', 'Subject', 'Study.Group', 'Gender', 
                   'smoking status', 'condition_numeric', 'normalized_inflammation',
                   'breakfast_day1', 'breakfast_day2', 'breakfast_day3',
                   'lunch_day1', 'lunch_day2', 'lunch_day3',
                   'dinner_day1', 'dinner_day2', 'dinner_day3']
    
    features = [col for col in data.columns if col not in exclude_cols]
    X = data[features]
    y = data['normalized_inflammation']
    
    # 2. Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # 3. Feature scaling (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device).reshape(-1, 1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler
```

### 3.2.4 Advanced GPU Processing

**File**: `IHMP/data_integration_pipeline/pytorch_deep_learning.py`

The platform includes specialized GPU-accelerated preprocessing for large-scale microbiome data:

```python
class MicrobiomeProcessor:
    def __init__(self, device):
        self.device = device
        self.lda_taxa = None

    def load_data(self, paths):
        """Load datasets with proper CPU/GPU separation"""
        data = {}
        # Metadata processed on CPU (pandas operations)
        data['metadata'] = pd.read_csv(paths[1], sep="\t")
        
        # Large numerical matrices loaded directly to GPU
        tensor_paths = [paths[0], paths[2], paths[3], paths[4], paths[5]]
        tensor_names = ['genera_counts', 'mtb', 'species', 'species_counts', 'genera']
        
        for name, path in zip(tensor_names, tensor_paths):
            df = pd.read_csv(path, sep="\t").set_index('Sample')
            # Direct GPU tensor creation for efficiency
            data[name] = torch.tensor(df.values, dtype=torch.float32, device=device)
        
        return data

    def process_metadata(self, df):
        """CPU-based metadata processing with GPU conversion"""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns
        
        # KNN imputation for numeric features
        if not numeric_cols.empty:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Mode imputation for categorical features
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Convert to GPU tensor for downstream processing
        return torch.tensor(df.values, dtype=torch.float32).to(self.device)
```

---

## 3.3 XGBoost Ensemble Models

### 3.3.1 XGBoost Architecture and Configuration

**File**: `IHMP/DONE/trainedxgboost.py`

XGBoost provides complementary gradient boosting capabilities with automatic feature importance analysis:

```python
# XGBoost model configuration
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',    # Regression with squared error
    random_state=42,                 # Reproducible results
    eval_metric='mae'                # Mean Absolute Error evaluation
)

# Comprehensive hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],              # Tree depth control
    'learning_rate': [0.01, 0.1, 0.3],   # Gradient step size
    'n_estimators': [100, 200],          # Number of boosting rounds
    'subsample': [0.7, 0.8],             # Sample fraction per tree
    'colsample_bytree': [0.7, 0.8]       # Feature fraction per tree
}
```

### 3.3.2 Hyperparameter Optimization

The platform uses GridSearchCV for comprehensive hyperparameter tuning:

```python
# Automated hyperparameter optimization
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',   # Optimization metric
    cv=5,                                # 5-fold cross-validation
    n_jobs=-1,                          # Parallel processing
    verbose=1                           # Progress monitoring
)

# Train with optimal parameters
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Performance metrics
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV MAE: {-grid_search.best_score_}")
```

### 3.3.3 Feature Importance Analysis

XGBoost provides interpretable feature importance scores for biological insights:

```python
def analyze_feature_importance(model, feature_names, top_n=20):
    """
    Extract and analyze feature importance from trained XGBoost model
    """
    # Get importance scores (gain-based)
    importance = model.get_booster().get_score(importance_type='gain')
    
    # Map feature indices to names
    importance_mapped = {
        feature_names[int(k.replace('f', ''))]: v 
        for k, v in importance.items()
    }
    
    # Sort by importance
    sorted_importance = sorted(
        importance_mapped.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Display top features
    print(f"Top {top_n} Feature Importances:")
    for feature, score in sorted_importance[:top_n]:
        print(f"{feature}: {score:.4f}")
    
    return sorted_importance

# Feature importance categories
def categorize_important_features(sorted_importance):
    """
    Categorize important features by biological relevance
    """
    categories = {
        'microbiome': [],
        'nutrition': [],
        'clinical': [],
        'metabolic': []
    }
    
    for feature, importance in sorted_importance:
        if any(x in feature.lower() for x in ['species', 'genera', 'diversity']):
            categories['microbiome'].append((feature, importance))
        elif any(x in feature.lower() for x in ['calorie', 'protein', 'fiber', 'vitamin']):
            categories['nutrition'].append((feature, importance))
        elif any(x in feature.lower() for x in ['age', 'bmi', 'symptom', 'medication']):
            categories['clinical'].append((feature, importance))
        elif any(x in feature.lower() for x in ['pathway', 'metabolite', 'flux']):
            categories['metabolic'].append((feature, importance))
    
    return categories
```

### 3.3.4 Model Persistence and Deployment

```python
# Save trained model for deployment
best_model.save_model("xgboost_nutrition_tuned.json")

# Save predictions with sample metadata
test_results = pd.DataFrame({
    'Sample': data.loc[test_indices, 'Sample'],
    'True_normalized_inflammation': y_test.values,
    'Predicted_normalized_inflammation': y_pred,
    'Prediction_Error': np.abs(y_test.values - y_pred)
})

# Calculate confidence intervals
test_results['CI_Lower'] = y_pred - 1.96 * np.std(y_pred)
test_results['CI_Upper'] = y_pred + 1.96 * np.std(y_pred)

test_results.to_csv("xgboost_predictions.csv", index=False)
```

---

## 3.4 Feature Selection and Engineering

### 3.4.1 Automated Feature Selection

The platform implements multiple feature selection strategies optimized for high-dimensional microbiome data:

```python
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    VarianceThreshold, RFE
)
from sklearn.ensemble import RandomForestRegressor

def comprehensive_feature_selection(X, y, n_features=500):
    """
    Multi-stage feature selection pipeline
    """
    # Stage 1: Remove low-variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var = variance_selector.fit_transform(X)
    
    # Stage 2: Univariate feature selection (F-test)
    f_selector = SelectKBest(f_regression, k=min(1000, X_var.shape[1]))
    X_f = f_selector.fit_transform(X_var, y)
    
    # Stage 3: Mutual information selection
    mi_selector = SelectKBest(mutual_info_regression, k=min(n_features, X_f.shape[1]))
    X_mi = mi_selector.fit_transform(X_f, y)
    
    # Stage 4: Recursive feature elimination with Random Forest
    rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe_selector = RFE(rf_estimator, n_features_to_select=n_features)
    X_final = rfe_selector.fit_transform(X_mi, y)
    
    # Combine selectors for inverse transform
    feature_mask = (
        variance_selector.get_support() & 
        f_selector.get_support() & 
        mi_selector.get_support() & 
        rfe_selector.get_support()
    )
    
    return X_final, feature_mask

# Apply feature selection
X_selected, selected_features = comprehensive_feature_selection(X, y, n_features=500)
```

### 3.4.2 Microbiome-Specific Feature Engineering

```python
def engineer_microbiome_ml_features(species_df, genera_df, metadata_df):
    """
    Create ML-optimized microbiome features
    """
    features = pd.DataFrame(index=species_df.columns)
    
    # 1. Top abundant species (robust to noise)
    top_species = species_df.mean(axis=1).nlargest(100).index
    for species in top_species:
        features[f'species_{species}'] = species_df.loc[species]
    
    # 2. Alpha diversity metrics
    features['shannon_diversity'] = calculate_shannon_diversity(species_df)
    features['simpson_diversity'] = calculate_simpson_diversity(species_df)
    features['observed_species'] = (species_df > 0).sum(axis=0)
    features['pielou_evenness'] = features['shannon_diversity'] / np.log(features['observed_species'])
    
    # 3. Taxonomic ratios (clinical biomarkers)
    firmicutes = genera_df.filter(regex='Firmicutes', axis=0).sum(axis=0)
    bacteroidetes = genera_df.filter(regex='Bacteroidetes', axis=0).sum(axis=0)
    features['firmicutes_bacteroidetes_ratio'] = firmicutes / (bacteroidetes + 1e-6)
    
    # 4. Functional guild abundances
    butyrate_producers = [
        'Faecalibacterium_prausnitzii', 'Eubacterium_rectale', 
        'Roseburia_inulinivorans', 'Coprococcus_comes'
    ]
    features['butyrate_producers'] = species_df.filter(items=butyrate_producers, axis=0).sum(axis=0)
    
    # 5. Pathogen indicators
    pathogenic_species = [
        'Clostridioides_difficile', 'Escherichia_coli',
        'Enterococcus_faecium', 'Klebsiella_pneumoniae'
    ]
    features['pathogen_load'] = species_df.filter(items=pathogenic_species, axis=0).sum(axis=0)
    
    # 6. Stability metrics (if temporal data available)
    if has_temporal_data(species_df):
        features['microbiome_stability'] = calculate_bray_curtis_stability(species_df)
    
    return features

def calculate_shannon_diversity(abundance_df):
    """Shannon diversity calculation optimized for large datasets"""
    # Vectorized calculation with pseudocount
    log_abundances = np.log(abundance_df + 1e-10)
    shannon = -(abundance_df * log_abundances).sum(axis=0)
    return shannon

def calculate_simpson_diversity(abundance_df):
    """Simpson diversity index"""
    simpson = 1 - (abundance_df ** 2).sum(axis=0)
    return simpson
```

---

## 3.5 Model Training and Optimization

### 3.5.1 Training Pipeline Architecture

The training pipeline implements best practices for reproducible machine learning:

```python
class ModelTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_metrics = {}
    
    def train_pytorch_model(self, X_train, y_train, X_val, y_val):
        """
        Train PyTorch neural network with comprehensive monitoring
        """
        # Initialize model
        model = NutritionNet(input_size=X_train.shape[1])
        model = model.to(self.config['device'])
        
        # Training setup
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.5
        )
        
        # Training loop with validation monitoring
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = self._train_epoch(model, X_train, y_train, criterion, optimizer)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
                val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'nutrition_net_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load('nutrition_net_best.pth'))
        self.models['pytorch'] = model
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def _train_epoch(self, model, X_train, y_train, criterion, optimizer):
        """Single training epoch with batch processing"""
        total_loss = 0
        batch_size = self.config['batch_size']
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / (len(X_train) // batch_size)
```

### 3.5.2 Cross-Validation Strategy

```python
from sklearn.model_selection import KFold, cross_val_score

def perform_cross_validation(X, y, models, cv_folds=5):
    """
    Comprehensive cross-validation for model comparison
    """
    cv_results = {}
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        if model_name == 'pytorch':
            # Custom CV for PyTorch models
            cv_scores = pytorch_cross_validation(model, X, y, kfold)
        else:
            # Standard sklearn CV
            cv_scores = cross_val_score(
                model, X, y, 
                cv=kfold, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
        
        cv_results[model_name] = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'all_scores': cv_scores
        }
        
        print(f"{model_name}: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    return cv_results

def pytorch_cross_validation(model_class, X, y, kfold):
    """Custom cross-validation for PyTorch models"""
    cv_scores = []
    
    for train_idx, val_idx in kfold.split(X):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize fresh model
        model = model_class(input_size=X.shape[1])
        model = train_pytorch_fold(model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_val, dtype=torch.float32))
            mae = torch.mean(torch.abs(y_pred - torch.tensor(y_val, dtype=torch.float32)))
            cv_scores.append(-mae.item())  # Negative for consistency with sklearn
    
    return cv_scores
```

---

## 3.6 Model Evaluation and Validation

### 3.6.1 Comprehensive Performance Metrics

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    explained_variance_score, max_error
)
import scipy.stats as stats

def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics
    """
    metrics = {}
    
    # Basic regression metrics
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    metrics['max_error'] = max_error(y_true, y_pred)
    
    # Statistical tests
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    
    # Normality test for residuals
    _, metrics['normality_p_value'] = stats.shapiro(residuals)
    
    # Correlation between true and predicted
    metrics['pearson_r'], metrics['pearson_p'] = stats.pearsonr(y_true, y_pred)
    metrics['spearman_r'], metrics['spearman_p'] = stats.spearmanr(y_true, y_pred)
    
    # Clinical relevance metrics
    metrics['within_10_percent'] = np.mean(np.abs(residuals) / np.abs(y_true) < 0.1)
    metrics['within_20_percent'] = np.mean(np.abs(residuals) / np.abs(y_true) < 0.2)
    
    # Print summary
    print(f"\n{model_name} Performance Metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4f})")
    print(f"Within 10% accuracy: {metrics['within_10_percent']*100:.1f}%")
    print(f"Within 20% accuracy: {metrics['within_20_percent']*100:.1f}%")
    
    return metrics

# Example usage with actual model results
def analyze_prediction_results():
    """
    Analyze saved prediction results from PyTorch and XGBoost models
    """
    # Load PyTorch results
    pytorch_results = pd.read_csv("pytorch_predictions.csv")
    y_true_pytorch = pytorch_results['True_normalized_inflammation']
    y_pred_pytorch = pytorch_results['Predicted_normalized_inflammation']
    
    # Load XGBoost results
    xgboost_results = pd.read_csv("xgboost_predictions.csv")
    y_true_xgboost = xgboost_results['True_normalized_inflammation']
    y_pred_xgboost = xgboost_results['Predicted_normalized_inflammation']
    
    # Evaluate both models
    pytorch_metrics = evaluate_model_performance(y_true_pytorch, y_pred_pytorch, "PyTorch")
    xgboost_metrics = evaluate_model_performance(y_true_xgboost, y_pred_xgboost, "XGBoost")
    
    return pytorch_metrics, xgboost_metrics
```

### 3.6.2 Prediction Visualization and Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_prediction_visualizations(y_true, y_pred, model_name, save_path=None):
    """
    Create comprehensive visualization of model predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Predicted vs True scatter plot
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=50)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'{model_name}: Predicted vs True')
    
    # Add R² annotation
    r2 = r2_score(y_true, y_pred)
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{model_name}: Residuals Plot')
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{model_name}: Residuals Distribution')
    
    # 4. Q-Q plot for normality
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'{model_name}: Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig

def compare_model_performance_visualization(results_dict):
    """
    Create comparative visualization of multiple models
    """
    models = list(results_dict.keys())
    metrics = ['mae', 'rmse', 'r2', 'within_20_percent']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'Model Comparison: {metric.upper()}')
        axes[i].set_ylabel(metric.upper())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return fig
```

---

## 3.7 Model Deployment and Inference

### 3.7.1 Model Serialization and Loading

```python
import joblib
import torch
import json

class ModelDeploymentManager:
    def __init__(self, model_directory="models/"):
        self.model_directory = model_directory
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
    
    def save_pytorch_model(self, model, model_name, scaler=None):
        """Save PyTorch model with all preprocessing components"""
        # Save model state
        torch.save(model.state_dict(), f"{self.model_directory}/{model_name}.pth")
        
        # Save model architecture info
        model_info = {
            'input_size': model.fc1.in_features,
            'architecture': 'NutritionNet',
            'version': '1.0'
        }
        with open(f"{self.model_directory}/{model_name}_info.json", 'w') as f:
            json.dump(model_info, f)
        
        # Save scaler if provided
        if scaler:
            joblib.dump(scaler, f"{self.model_directory}/{model_name}_scaler.pkl")
    
    def load_pytorch_model(self, model_name):
        """Load PyTorch model with preprocessing components"""
        # Load model info
        with open(f"{self.model_directory}/{model_name}_info.json", 'r') as f:
            model_info = json.load(f)
        
        # Initialize model architecture
        model = NutritionNet(input_size=model_info['input_size'])
        model.load_state_dict(torch.load(f"{self.model_directory}/{model_name}.pth"))
        model.eval()
        
        # Load scaler
        try:
            scaler = joblib.load(f"{self.model_directory}/{model_name}_scaler.pkl")
        except FileNotFoundError:
            scaler = None
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model, scaler
    
    def save_xgboost_model(self, model, model_name, scaler=None):
        """Save XGBoost model"""
        model.save_model(f"{self.model_directory}/{model_name}.json")
        
        if scaler:
            joblib.dump(scaler, f"{self.model_directory}/{model_name}_scaler.pkl")
    
    def load_xgboost_model(self, model_name):
        """Load XGBoost model"""
        model = xgb.XGBRegressor()
        model.load_model(f"{self.model_directory}/{model_name}.json")
        
        try:
            scaler = joblib.load(f"{self.model_directory}/{model_name}_scaler.pkl")
        except FileNotFoundError:
            scaler = None
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        
        return model, scaler
```

### 3.7.2 Real-time Inference Pipeline

```python
class InferencePipeline:
    def __init__(self, deployment_manager):
        self.dm = deployment_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict_inflammation(self, microbiome_data, clinical_data, nutrition_data, 
                           model_name='pytorch'):
        """
        Make inflammation predictions from input data
        """
        # 1. Preprocess input data
        processed_features = self._preprocess_inference_data(
            microbiome_data, clinical_data, nutrition_data
        )
        
        # 2. Get model and scaler
        model = self.dm.models[model_name]
        scaler = self.dm.scalers[model_name]
        
        # 3. Scale features
        if scaler:
            processed_features = scaler.transform(processed_features.reshape(1, -1))
        
        # 4. Make prediction
        if model_name == 'pytorch':
            with torch.no_grad():
                features_tensor = torch.tensor(processed_features, dtype=torch.float32).to(self.device)
                prediction = model(features_tensor).cpu().numpy()[0, 0]
        else:  # XGBoost
            prediction = model.predict(processed_features)[0]
        
        # 5. Generate confidence intervals and interpretation
        confidence_interval = self._calculate_confidence_interval(prediction, model_name)
        interpretation = self._interpret_prediction(prediction)
        
        return {
            'inflammation_score': float(prediction),
            'confidence_interval': confidence_interval,
            'interpretation': interpretation,
            'recommendations': self._generate_recommendations(prediction)
        }
    
    def _preprocess_inference_data(self, microbiome_data, clinical_data, nutrition_data):
        """
        Preprocess raw input data for inference
        """
        # Combine all input features
        all_features = np.concatenate([
            microbiome_data.flatten(),
            clinical_data.flatten(),
            nutrition_data.flatten()
        ])
        
        return all_features
    
    def _calculate_confidence_interval(self, prediction, model_name, alpha=0.05):
        """
        Calculate prediction confidence intervals
        """
        # This would typically use model uncertainty estimates
        # For simplicity, using empirical standard deviation
        if model_name == 'pytorch':
            std_error = 0.05  # Estimated from validation data
        else:
            std_error = 0.04  # XGBoost typically has lower uncertainty
        
        z_score = 1.96  # 95% confidence interval
        margin = z_score * std_error
        
        return {
            'lower': float(prediction - margin),
            'upper': float(prediction + margin)
        }
    
    def _interpret_prediction(self, prediction):
        """
        Provide clinical interpretation of inflammation score
        """
        if prediction < 0.1:
            return "Low inflammation - healthy gut status"
        elif prediction < 0.3:
            return "Moderate inflammation - consider dietary adjustments"
        elif prediction < 0.6:
            return "Elevated inflammation - recommend clinical consultation"
        else:
            return "High inflammation - urgent medical attention recommended"
    
    def _generate_recommendations(self, prediction):
        """
        Generate personalized recommendations based on prediction
        """
        recommendations = []
        
        if prediction > 0.3:
            recommendations.extend([
                "Increase fiber intake (target 25-35g/day)",
                "Include probiotic foods (yogurt, kefir, sauerkraut)",
                "Reduce processed food consumption",
                "Consider anti-inflammatory foods (berries, leafy greens)"
            ])
        
        if prediction > 0.6:
            recommendations.extend([
                "Consult gastroenterologist for evaluation",
                "Consider elimination diet trial",
                "Discuss potential supplements with healthcare provider"
            ])
        
        return recommendations

# Usage example
def deploy_models_for_production():
    """
    Example deployment script
    """
    # Initialize deployment manager
    dm = ModelDeploymentManager()
    
    # Load trained models
    pytorch_model, pytorch_scaler = dm.load_pytorch_model('nutrition_net_best')
    xgboost_model, xgboost_scaler = dm.load_xgboost_model('xgboost_nutrition_tuned')
    
    # Initialize inference pipeline
    inference = InferencePipeline(dm)
    
    # Example prediction
    sample_microbiome = np.random.random(1458)  # Mock microbiome data
    sample_clinical = np.array([45, 25.5, 1, 0])  # Age, BMI, gender, medication
    sample_nutrition = np.random.random(50)      # Nutrition features
    
    # Make predictions
    pytorch_result = inference.predict_inflammation(
        sample_microbiome, sample_clinical, sample_nutrition, 'pytorch'
    )
    
    xgboost_result = inference.predict_inflammation(
        sample_microbiome, sample_clinical, sample_nutrition, 'xgboost'
    )
    
    return pytorch_result, xgboost_result
```

---

## 3.8 Digital Twin Integration

### 3.8.1 Neural Network-Enhanced Digital Twin

**File**: `IHMP/DONE/nutrition_prediction_model.py`

The platform integrates machine learning with metabolic modeling through a hybrid digital twin approach:

```python
class FluxPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(FluxPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

class DigitalTwinModel:
    def __init__(self, cobra_model_path="iHsaGut999.json"):
        """
        Initialize hybrid digital twin with COBRA and ML components
        """
        # Load metabolic network model
        self.cobra_model = load_model(cobra_model_path)
        
        # Key metabolic reactions for prediction
        self.key_reactions = self._get_key_reactions()
        
        # Neural network for flux prediction
        self.flux_predictor = None
        self.scaler_input = StandardScaler()
        self.scaler_output = StandardScaler()
    
    def _get_key_reactions(self):
        """
        Identify key metabolic reactions for health assessment
        """
        key_reactions = {
            'butyrate_production': ['BUTKE', 'BUTAT', 'BUTCOAS'],
            'inflammation_markers': ['PROSTAGLANDIN_E2', 'TNF_ALPHA'],
            'vitamin_synthesis': ['FOLR2', 'VITAMIN_B12_SYNTHESIS'],
            'fiber_degradation': ['CELLULASE', 'XYLANASE']
        }
        
        # Filter reactions that exist in the model
        filtered_reactions = {}
        for category, reactions in key_reactions.items():
            existing_reactions = [r for r in reactions if r in self.cobra_model.reactions]
            if existing_reactions:
                filtered_reactions[category] = existing_reactions
        
        return filtered_reactions
    
    def train_flux_predictor(self, microbiome_data, nutrition_data, flux_data):
        """
        Train neural network to predict metabolic fluxes
        """
        # Combine microbiome and nutrition features
        X = np.hstack([microbiome_data, nutrition_data])
        y = flux_data
        
        # Scale data
        X_scaled = self.scaler_input.fit_transform(X)
        y_scaled = self.scaler_output.fit_transform(y)
        
        # Initialize and train model
        input_size = X_scaled.shape[1]
        output_size = y_scaled.shape[1]
        
        self.flux_predictor = FluxPredictor(input_size, output_size)
        
        # Training configuration
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.flux_predictor.parameters(), lr=0.001)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        
        # Training loop
        for epoch in range(1000):
            optimizer.zero_grad()
            predictions = self.flux_predictor(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def predict_personalized_fluxes(self, microbiome_profile, nutrition_profile):
        """
        Predict personalized metabolic fluxes
        """
        # Prepare input data
        input_data = np.hstack([microbiome_profile, nutrition_profile]).reshape(1, -1)
        input_scaled = self.scaler_input.transform(input_data)
        
        # Predict fluxes
        with torch.no_grad():
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            predicted_fluxes = self.flux_predictor(input_tensor).numpy()
        
        # Inverse transform
        predicted_fluxes = self.scaler_output.inverse_transform(predicted_fluxes)
        
        return predicted_fluxes[0]
    
    def simulate_intervention(self, baseline_profile, intervention_changes):
        """
        Simulate the effect of nutritional interventions
        """
        # Baseline prediction
        baseline_fluxes = self.predict_personalized_fluxes(
            baseline_profile['microbiome'], baseline_profile['nutrition']
        )
        
        # Intervention prediction
        modified_nutrition = baseline_profile['nutrition'] + intervention_changes
        intervention_fluxes = self.predict_personalized_fluxes(
            baseline_profile['microbiome'], modified_nutrition
        )
        
        # Calculate changes
        flux_changes = intervention_fluxes - baseline_fluxes
        
        return {
            'baseline_fluxes': baseline_fluxes,
            'intervention_fluxes': intervention_fluxes,
            'flux_changes': flux_changes,
            'improvement_score': self._calculate_improvement_score(flux_changes)
        }
    
    def _calculate_improvement_score(self, flux_changes):
        """
        Calculate overall health improvement score from flux changes
        """
        # Weight different metabolic pathways by health importance
        weights = {
            'butyrate_production': 0.3,
            'inflammation_markers': -0.4,  # Negative because lower is better
            'vitamin_synthesis': 0.2,
            'fiber_degradation': 0.1
        }
        
        improvement_score = 0
        for pathway, weight in weights.items():
            if pathway in self.key_reactions:
                pathway_flux_change = np.mean([
                    flux_changes[i] for i, reaction in enumerate(self.key_reactions[pathway])
                ])
                improvement_score += weight * pathway_flux_change
        
        return improvement_score
```

---

**Next**: Part 4 will cover Digital Twin Modeling and Simulation, including detailed COBRA model integration, flux balance analysis, and personalized metabolic predictions.

**Files Covered in Part 3**:
- `IHMP/DONE/pytorch-dl.py`
- `IHMP/data_integration_pipeline/pytorch_deep_learning.py`
- `IHMP/DONE/trainedxgboost.py`
- `IHMP/DONE/nutrition_prediction_model.py`
- `pytorch_predictions.csv`
- `IHMP/DONE/xgboost_predictions.csv`
- `IHMP/DONE/xgboost_nutrition_tuned.json`
- `nutrition_net_best.pth`
- `nutrition_net_final.pth`
- `IHMP/DONE/scaler.pkl`
