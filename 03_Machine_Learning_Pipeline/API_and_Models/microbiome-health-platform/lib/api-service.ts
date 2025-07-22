"use client";

const API_URL = 'http://localhost:8000';
const API_KEY = 'your-secret-api-key'; // In production, this should be stored securely
const USE_MOCK_DATA = true; // Set to false to use the actual API

// Types for API requests and responses
export interface SymptomInput {
  bloating: number;
  abdominal_pain: number;
  diarrhea: number;
  constipation: number;
}

export interface FoodIntake {
  food_name: string;
  fiber_g: number;
  prebiotic_score: number;
}

export interface UserInput {
  age: number;
  weight: number;
  height: number;
  diet_type: string;
  calories_target: number;
  symptoms: SymptomInput;
  recent_foods: FoodIntake[];
}

export interface PredictionInput {
  features: Record<string, number>;
  return_feature_importance?: boolean;
}

export interface DigitalTwinInput {
  patient_id: string;
  age: number;
  weight: number;
  height: number;
  daily_fiber: number;
  calories_intake: number;
  symptoms: SymptomInput;
  microbiome_diversity?: number;
  calprotectin?: number;
  additional_features?: Record<string, number>;
}

export interface TreatmentOption {
  name: string;
  type: string;
  estimated_effect: Record<string, number>;
}

export interface TreatmentInput {
  patient_id: string;
  current_metrics: Record<string, number>;
  available_treatments: TreatmentOption[];
  treatment_count?: number;
}

// Mock responses for testing
const MOCK_RESPONSES = {
  digitalTwin: {
    butyrate_flux: 0.43,
    inflammation_score: 1.2,
    metabolic_health_score: 78.5,
    gut_permeability_estimate: 0.35,
    recommendations: [
      "Increase dietary fiber intake to at least 25g per day",
      "Consider anti-inflammatory diet with omega-3 fatty acids",
      "Add fermented foods to your diet for probiotic benefits"
    ],
    risk_factors: [
      "Moderate inflammatory markers",
      "Reduced microbiome diversity"
    ],
    simulation_confidence: 0.82
  },
  treatmentOptimize: {
    recommended_treatments: [
      {
        name: "High-fiber Diet",
        type: "diet",
        estimated_effect: {
          inflammation: -0.3,
          bloating: -1,
          gut_permeability: -0.1,
          butyrate_production: 0.2,
          microbiome_diversity: 0.1
        }
      },
      {
        name: "Probiotic (B. longum)",
        type: "supplement",
        estimated_effect: {
          inflammation: -0.2,
          bloating: -1,
          gut_permeability: -0.15,
          butyrate_production: 0.1,
          microbiome_diversity: 0.15
        }
      },
      {
        name: "Anti-inflammatory (Omega-3)",
        type: "supplement",
        estimated_effect: {
          inflammation: -0.3,
          bloating: 0,
          gut_permeability: -0.05,
          butyrate_production: 0,
          microbiome_diversity: 0
        }
      }
    ],
    predicted_improvement: {
      inflammation: 0.8,
      bloating: 2.0,
      gut_permeability: 0.3,
      butyrate_production: 0.3,
      microbiome_diversity: 0.25
    },
    confidence: 0.78
  }
};

// API service class
class ApiService {
  private async fetchWithAuth(endpoint: string, options: RequestInit = {}) {
    const headers = {
      ...options.headers,
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY,
    };

    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.detail || `API request failed with status ${response.status}`);
    }

    return response.json();
  }

  // XGBoost prediction endpoint
  async predictXgboost(input: PredictionInput) {
    if (USE_MOCK_DATA) {
      return Promise.resolve({
        prediction: 0.75,
        feature_importance: {
          "avg_daily_fiber": 0.42,
          "bloating": 0.23,
          "constipation": 0.18,
          "age": 0.12,
          "bmi": 0.05
        }
      });
    }
    
    return this.fetchWithAuth('/predict/xgboost', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  }

  // Digital twin simulation endpoint
  async simulateDigitalTwin(input: DigitalTwinInput) {
    if (USE_MOCK_DATA) {
      return Promise.resolve(MOCK_RESPONSES.digitalTwin);
    }
    
    return this.fetchWithAuth('/digital-twin/simulate', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  }

  // Treatment optimization endpoint
  async optimizeTreatment(input: TreatmentInput) {
    if (USE_MOCK_DATA) {
      return Promise.resolve(MOCK_RESPONSES.treatmentOptimize);
    }
    
    return this.fetchWithAuth('/treatment/optimize', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  }

  // Patient meal plan endpoint
  async generateMealPlan(input: UserInput) {
    if (USE_MOCK_DATA) {
      return Promise.resolve({
        meal_plan: [],
        fiber_target: 30,
        butyrate_score: 0.7,
        inflammation_score: 1.2
      });
    }
    
    return this.fetchWithAuth('/patient/meal-plan', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  }

  // Doctor analysis endpoint
  async getDoctorAnalysis(input: { 
    patient_id: string;
    calprotectin?: number;
    microbiome_diversity?: string;
    features: Record<string, number>;
  }) {
    if (USE_MOCK_DATA) {
      return Promise.resolve({
        inflammation_risk: 0.65,
        butyrate_flux: 0.4,
        recommendations: [
          "Implement prebiotic and probiotic intervention",
          "Consider anti-inflammatory diet"
        ]
      });
    }
    
    return this.fetchWithAuth('/doctor/analysis', {
      method: 'POST',
      body: JSON.stringify(input),
    });
  }

  // Health check endpoint
  async healthCheck() {
    if (USE_MOCK_DATA) {
      return Promise.resolve({ status: "healthy" });
    }
    
    return this.fetchWithAuth('/health');
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService; 