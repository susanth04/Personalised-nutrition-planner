/*
This file is for documentation purposes only.
It describes the database schema for the Microbiome Health Platform.

Tables:
1. profiles - User profiles with role information
2. uploads - Microbiome data file uploads
3. symptoms - Patient symptom logs
4. recommendations - Doctor recommendations to patients
5. analysis_results - Results of microbiome data analysis
*/

// Profiles table
interface Profile {
  id: string // Primary key, matches auth.users.id
  email: string
  name: string
  role: "patient" | "doctor"
  created_at: string
  updated_at: string
}

// Uploads table
interface Upload {
  id: string // Primary key
  user_id: string // Foreign key to profiles.id
  file_name: string
  file_path: string
  file_url: string
  description: string
  file_type: string
  file_size: number
  created_at: string
  updated_at: string
}

// Symptoms table
interface Symptom {
  id: string // Primary key
  user_id: string // Foreign key to profiles.id
  symptoms: string[] // Array of symptom names
  severity: number // 1-10 scale
  notes: string
  date: string
  created_at: string
  updated_at: string
}

// Recommendations table
interface Recommendation {
  id: string // Primary key
  doctor_id: string // Foreign key to profiles.id (doctor)
  patient_id: string // Foreign key to profiles.id (patient)
  recommendation: string
  created_at: string
  updated_at: string
}

// Analysis results table
interface AnalysisResult {
  id: string // Primary key
  user_id: string // Foreign key to profiles.id
  upload_id: string // Foreign key to uploads.id
  health_score: number
  diversity_index: number
  bacteria_composition: Record<string, number> // JSON object with bacteria names and percentages
  analysis_type: string
  analysis_data: any // JSON object with detailed analysis results
  created_at: string
  updated_at: string
}
