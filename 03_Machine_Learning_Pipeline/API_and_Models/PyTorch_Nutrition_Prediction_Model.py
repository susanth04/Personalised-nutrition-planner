import pandas as pd
import numpy as np
import cobra
from cobra.io import load_model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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
        Initialize the Digital Twin model with a COBRA metabolic model
        and a neural network to predict fluxes from microbiome and diet data.
        
        Args:
            cobra_model_path: Path to the COBRA model JSON file
        """
        try:
            # Load the COBRA model
            print(f"Loading COBRA model from {cobra_model_path}...")
            self.cobra_model = load_model(cobra_model_path)
            print(f"Model loaded with {len(self.cobra_model.reactions)} reactions and {len(self.cobra_model.metabolites)} metabolites")
            
            # Get key reaction IDs we're interested in
            self.key_reactions = self._get_key_reactions()
            
            # Define neural network for flux prediction
            self.flux_predictor = None
            self.scaler_input = StandardScaler()
            self.scaler_output = StandardScaler()
            
        except Exception as e:
            print(f"Error initializing Digital Twin model: {e}")
            # Fallback to a minimal model if the specified one fails
            self.cobra_model = cobra.Model('minimal_gut_model')
            self._create_minimal_model()
    
    def _create_minimal_model(self):
        """Create a minimal gut model if loading fails"""
        print("Creating minimal gut model...")
        
        # Add metabolites
        acetate = cobra.Metabolite('acetate_c', formula='C2H4O2', name='Acetate')
        butyrate = cobra.Metabolite('butyrate_c', formula='C4H8O2', name='Butyrate')
        propionate = cobra.Metabolite('propionate_c', formula='C3H6O2', name='Propionate')
        fiber = cobra.Metabolite('fiber_c', formula='C6H10O5', name='Dietary Fiber')
        
        # External metabolites
        acetate_e = cobra.Metabolite('acetate_e', formula='C2H4O2', name='External Acetate')
        butyrate_e = cobra.Metabolite('butyrate_e', formula='C4H8O2', name='External Butyrate')
        propionate_e = cobra.Metabolite('propionate_e', formula='C3H6O2', name='External Propionate')
        fiber_e = cobra.Metabolite('fiber_e', formula='C6H10O5', name='External Dietary Fiber')
        
        # Add reactions
        # Fiber uptake
        r1 = cobra.Reaction('EX_fiber')
        r1.name = 'Fiber Exchange'
        r1.lower_bound = -10  # Allow uptake
        r1.upper_bound = 0    # No secretion
        r1.add_metabolites({fiber_e: -1, fiber_c: 1})
        
        # Fiber to SCFAs conversion reactions
        r2 = cobra.Reaction('FIBER_TO_ACETATE')
        r2.name = 'Fiber to Acetate'
        r2.lower_bound = 0
        r2.upper_bound = 100
        r2.add_metabolites({fiber_c: -1, acetate_c: 2})  # 1 fiber -> 2 acetate
        
        r3 = cobra.Reaction('FIBER_TO_BUTYRATE')
        r3.name = 'Fiber to Butyrate'
        r3.lower_bound = 0
        r3.upper_bound = 100
        r3.add_metabolites({fiber_c: -1, butyrate_c: 1})
        
        r4 = cobra.Reaction('FIBER_TO_PROPIONATE')
        r4.name = 'Fiber to Propionate'
        r4.lower_bound = 0
        r4.upper_bound = 100
        r4.add_metabolites({fiber_c: -1, propionate_c: 1})
        
        # SCFA exchange reactions
        r5 = cobra.Reaction('EX_acetate')
        r5.name = 'Acetate Exchange'
        r5.lower_bound = 0  # Only secretion
        r5.upper_bound = 100
        r5.add_metabolites({acetate_c: -1, acetate_e: 1})
        
        r6 = cobra.Reaction('EX_butyrate')
        r6.name = 'Butyrate Exchange'
        r6.lower_bound = 0  # Only secretion
        r6.upper_bound = 100
        r6.add_metabolites({butyrate_c: -1, butyrate_e: 1})
        
        r7 = cobra.Reaction('EX_propionate')
        r7.name = 'Propionate Exchange'
        r7.lower_bound = 0  # Only secretion
        r7.upper_bound = 100
        r7.add_metabolites({propionate_c: -1, propionate_e: 1})
        
        # Add all reactions to the model
        self.cobra_model.add_reactions([r1, r2, r3, r4, r5, r6, r7])
        
        # Set objective
        self.cobra_model.objective = 'EX_butyrate'  # Maximize butyrate production
        
        print(f"Created minimal model with {len(self.cobra_model.reactions)} reactions and {len(self.cobra_model.metabolites)} metabolites")
        
        # Define key reactions
        self.key_reactions = {'EX_fiber', 'FIBER_TO_ACETATE', 'FIBER_TO_BUTYRATE', 
                             'FIBER_TO_PROPIONATE', 'EX_acetate', 'EX_butyrate', 'EX_propionate'}
    
    def _get_key_reactions(self):
        """
        Get key reactions related to gut metabolism that we want to track
        """
        # Look for important reactions in the model
        key_patterns = [
            'EX_fiber', 'EX_butyrate', 'EX_acetate', 'EX_propionate',  # Exchange reactions
            'FIBER', 'SCFA', 'BUTYRATE', 'ACETATE', 'PROPIONATE',       # Production reactions
            'bile', 'BileAcid', 'BILE',                                # Bile acid metabolism
            'inflam', 'TNF', 'IL', 'cytokine',                         # Inflammation markers
            'LPS', 'flagellin', 'endotoxin'                            # Bacterial components
        ]
        
        # Find reactions matching patterns
        key_reactions = set()
        for pattern in key_patterns:
            for reaction in self.cobra_model.reactions:
                if pattern.lower() in reaction.id.lower() or pattern.lower() in reaction.name.lower():
                    key_reactions.add(reaction.id)
        
        # If no matches are found, use some common reactions as fallback
        if not key_reactions:
            try:
                # Try to find some common gut-related reactions
                common_ids = ['EX_glc__D_e', 'EX_ac_e', 'EX_but_e', 'EX_prop_e', 'EX_co2_e']
                for rxn_id in common_ids:
                    if rxn_id in self.cobra_model.reactions:
                        key_reactions.add(rxn_id)
            except:
                # Fallback to first 10 reactions if nothing works
                for rxn in list(self.cobra_model.reactions)[:10]:
                    key_reactions.add(rxn.id)
        
        print(f"Identified {len(key_reactions)} key reactions for tracking")
        return key_reactions
    
    def train_flux_predictor(self, X, y=None, epochs=100, batch_size=32):
        """
        Train the neural network to predict reaction fluxes from microbiome and nutrition data.
        
        Args:
            X: DataFrame with microbiome and nutrition features
            y: Optional DataFrame with known fluxes (if available from experimental data)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        print("Training flux predictor neural network...")
        
        # Prepare input data
        X_scaled = self.scaler_input.fit_transform(X)
        input_size = X_scaled.shape[1]
        
        # If we don't have actual flux data, generate synthetic training data
        # using the model by varying inputs slightly and seeing how fluxes change
        if y is None:
            print("No flux data provided, generating synthetic training data...")
            # Generate synthetic data by perturbing bounds and solving the model
            n_samples = len(X)
            output_size = len(self.key_reactions)
            y_synthetic = np.zeros((n_samples, output_size))
            
            # For each sample, slightly modify the model based on the data and solve
            for i in range(n_samples):
                # Use row data to modify reaction bounds
                self._modify_model_bounds_from_data(X.iloc[i])
                
                # Solve the model
                try:
                    solution = self.cobra_model.optimize()
                    # Extract fluxes for key reactions
                    for j, rxn_id in enumerate(self.key_reactions):
                        if rxn_id in self.cobra_model.reactions:
                            y_synthetic[i, j] = solution.fluxes.get(rxn_id, 0)
                except:
                    # If optimization fails, use zeros
                    print(f"Optimization failed for sample {i}")
            
            # Scale the output data
            y_scaled = self.scaler_output.fit_transform(y_synthetic)
        else:
            # Use provided flux data if available
            output_size = y.shape[1]
            y_scaled = self.scaler_output.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.flux_predictor = FluxPredictor(input_size, output_size)
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(self.flux_predictor.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            self.flux_predictor.train()
            running_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.flux_predictor(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
        
        print("Neural network training complete")
        
        # Save the model
        torch.save(self.flux_predictor.state_dict(), "flux_predictor.pth")
        print("Flux predictor model saved as flux_predictor.pth")
    
    def _modify_model_bounds_from_data(self, data_row):
        """
        Modify reaction bounds in the COBRA model based on data
        
        Args:
            data_row: Series with feature values for one sample
        """
        # Adjust fiber uptake based on dietary fiber
        if 'avg_daily_fiber' in data_row:
            fiber_uptake = float(data_row['avg_daily_fiber'])
            fiber_rx_id = None
            
            # Try different possible reaction IDs for fiber uptake
            for rx_id in ['EX_fiber', 'EX_fiber_e', 'EX_starch_e', 'EX_fib_e']:
                if rx_id in self.cobra_model.reactions:
                    fiber_rx_id = rx_id
                    break
            
            if fiber_rx_id:
                # Set lower bound to negative fiber intake (uptake)
                try:
                    self.cobra_model.reactions.get_by_id(fiber_rx_id).lower_bound = -max(0.1, fiber_uptake)
                except:
                    pass
        
        # Adjust protein reactions based on protein intake
        if 'avg_daily_protein' in data_row:
            protein_value = float(data_row['avg_daily_protein'])
            
            # Find protein-related reactions
            for reaction in self.cobra_model.reactions:
                if 'protein' in reaction.id.lower() or 'prot' in reaction.id.lower():
                    if 'EX_' in reaction.id:  # Exchange reaction
                        try:
                            reaction.lower_bound = -max(0.1, protein_value/10)
                        except:
                            pass
        
        # Adjust fat reactions based on fat intake
        if 'avg_daily_fat' in data_row:
            fat_value = float(data_row['avg_daily_fat'])
            
            # Find fat-related reactions
            for reaction in self.cobra_model.reactions:
                if 'fat' in reaction.id.lower() or 'lipid' in reaction.id.lower():
                    if 'EX_' in reaction.id:  # Exchange reaction
                        try:
                            reaction.lower_bound = -max(0.1, fat_value/10)
                        except:
                            pass
        
        # Modify reactions based on microbiome PC values
        for col in data_row.index:
            if col.startswith('mtb_PC') or col.startswith('species_PC'):
                pc_value = float(data_row[col])
                
                # Use PC values to modify reaction bounds
                # Higher values might indicate higher activity
                if abs(pc_value) > 0.1:  # Only consider significant PCs
                    # Adjust some random reactions based on PC value
                    # This is a simplified approach; in reality would need mapping from PCs to reactions
                    for i, rxn_id in enumerate(self.key_reactions):
                        if i % 5 == int(col.split('_')[1][2:]) % 5:  # Use PC number to select reactions
                            try:
                                reaction = self.cobra_model.reactions.get_by_id(rxn_id)
                                if pc_value > 0:
                                    # Increase upper bound for positive PC values
                                    reaction.upper_bound = min(1000, reaction.upper_bound * (1 + abs(pc_value)))
                                else:
                                    # Decrease upper bound for negative PC values
                                    reaction.upper_bound = max(0, reaction.upper_bound * (1 - abs(pc_value)/2))
                            except:
                                pass
    
    def predict_metabolites(self, data_row):
        """
        Predict metabolite production/consumption based on a single data row
        
        Args:
            data_row: Series with feature values for one sample
            
        Returns:
            dict: Dictionary with predicted metabolite values
        """
        # Modify model bounds based on data
        self._modify_model_bounds_from_data(data_row)
        
        # If neural network exists, use it to predict fluxes
        if self.flux_predictor is not None:
            try:
                # Prepare input data
                X = pd.DataFrame([data_row])
                X_scaled = self.scaler_input.transform(X)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                
                # Predict fluxes
                self.flux_predictor.eval()
                with torch.no_grad():
                    predicted_fluxes_scaled = self.flux_predictor(X_tensor).numpy()
                
                # Inverse transform to get actual flux values
                predicted_fluxes = self.scaler_output.inverse_transform(predicted_fluxes_scaled)
                
                # Set predicted fluxes in the model
                for i, rxn_id in enumerate(self.key_reactions):
                    if rxn_id in self.cobra_model.reactions:
                        # Set predicted flux as constraint
                        try:
                            rxn = self.cobra_model.reactions.get_by_id(rxn_id)
                            flux_value = predicted_fluxes[0, i]
                            # Allow some flexibility by setting bounds around predicted value
                            rxn.lower_bound = max(rxn.lower_bound, flux_value * 0.9)
                            rxn.upper_bound = min(rxn.upper_bound, flux_value * 1.1)
                        except:
                            pass
            except Exception as e:
                print(f"Error using neural network predictions: {e}")
                # Fall back to direct FBA if neural network fails
                pass
        
        # Solve the model with current constraints
        try:
            solution = self.cobra_model.optimize()
            
            # Extract key metabolites
            results = {}
            
            # Short-chain fatty acids (SCFAs)
            scfa_patterns = ['butyrate', 'acetate', 'propionate']
            for pattern in scfa_patterns:
                for rxn in self.cobra_model.reactions:
                    if f'EX_{pattern}' in rxn.id:
                        results[f'{pattern}_production'] = solution.fluxes.get(rxn.id, 0)
            
            # Add inflammation prediction based on butyrate (simplified)
            # High butyrate is generally associated with lower inflammation
            butyrate_flux = results.get('butyrate_production', 0)
            results['predicted_inflammation'] = max(0, 1 - (butyrate_flux / 10))
            
            return results
            
        except Exception as e:
            print(f"FBA optimization failed: {e}")
            # Return default values if optimization fails
            return {
                'butyrate_production': 0,
                'acetate_production': 0,
                'propionate_production': 0,
                'predicted_inflammation': 0.5  # Neutral prediction
            }
    
    def batch_predict(self, data_df):
        """
        Make predictions for a batch of data
        
        Args:
            data_df: DataFrame with samples
            
        Returns:
            DataFrame: Predictions for each sample
        """
        results = []
        for _, row in data_df.iterrows():
            results.append(self.predict_metabolites(row))
        
        return pd.DataFrame(results)

    def save_model(self, filepath="digital_twin_model.pkl"):
        """
        Save the model to a file
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'flux_predictor_state': self.flux_predictor.state_dict() if self.flux_predictor else None,
                'scaler_input': self.scaler_input,
                'scaler_output': self.scaler_output,
                'key_reactions': self.key_reactions
            }, f)
        print(f"Digital twin model saved to {filepath}")
    
    def load_model(self, filepath="digital_twin_model.pkl"):
        """
        Load the model from a file
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Load flux predictor if it exists
        if data['flux_predictor_state']:
            # Need to initialize with correct dimensions first
            input_size = data['scaler_input'].n_features_in_
            output_size = len(data['key_reactions'])
            self.flux_predictor = FluxPredictor(input_size, output_size)
            self.flux_predictor.load_state_dict(data['flux_predictor_state'])
            
        self.scaler_input = data['scaler_input']
        self.scaler_output = data['scaler_output']
        self.key_reactions = data['key_reactions']
        print(f"Digital twin model loaded from {filepath}")


# Example usage function
def example_usage():
    # Sample data loading
    data = pd.read_csv("sample_data.tsv", sep='\t')
    
    # Create and train the digital twin model
    dt_model = DigitalTwinModel()
    
    # Features for training
    feature_cols = [col for col in data.columns if col not in ['Dataset', 'Sample', 'Subject', 'Study.Group', 
                                                            'Gender', 'smoking status', 'condition_numeric', 
                                                            'normalized_inflammation', 'breakfast_day1', 
                                                            'breakfast_day2', 'breakfast_day3', 'lunch_day1', 
                                                            'lunch_day2', 'lunch_day3', 'dinner_day1', 
                                                            'dinner_day2', 'dinner_day3']]
    
    # Train the model
    X_train = data[feature_cols]
    dt_model.train_flux_predictor(X_train, epochs=50)
    
    # Make predictions
    predictions = dt_model.batch_predict(data[feature_cols])
    
    # Combine original data with predictions
    results = pd.concat([data[['Sample', 'normalized_inflammation']], predictions], axis=1)
    
    # Save results
    results.to_csv("digital_twin_predictions.csv", index=False)
    
    # Save the model for future use
    dt_model.save_model()
    
    return results

# If this script is run directly
if __name__ == "__main__":
    print("Running Digital Twin model example...")
    results = example_usage()
    print("Complete. Results saved to digital_twin_predictions.csv")