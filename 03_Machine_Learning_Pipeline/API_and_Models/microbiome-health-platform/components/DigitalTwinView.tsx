"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2, RefreshCw } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import apiService, { DigitalTwinInput } from "@/lib/api-service";

interface DigitalTwinProps {
  patientId: string;
  patientData?: {
    age: number;
    weight: number;
    height: number;
    symptoms: {
      bloating: number;
      abdominal_pain: number;
      diarrhea: number;
      constipation: number;
    }
  }
}

export default function DigitalTwinView({ patientId, patientData }: DigitalTwinProps) {
  const [loading, setLoading] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [twinData, setTwinData] = useState<any>(null);

  const runSimulation = async () => {
    if (!patientData) {
      setError("Patient data not available");
      return;
    }

    setSimulating(true);
    setError(null);

    try {
      // Create input for digital twin simulation
      const twinInput: DigitalTwinInput = {
        patient_id: patientId,
        age: patientData.age,
        weight: patientData.weight,
        height: patientData.height,
        daily_fiber: 25, // Default value, could be fetched from patient records
        calories_intake: 2000, // Default value, could be fetched from patient records
        symptoms: patientData.symptoms,
        // Optional fields if available
        microbiome_diversity: 0.65, // Example value
        calprotectin: 120, // Example value
      };

      const result = await apiService.simulateDigitalTwin(twinInput);
      setTwinData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run digital twin simulation");
    } finally {
      setSimulating(false);
    }
  };

  // Initialize simulation on component mount if patient data is available
  useEffect(() => {
    if (patientData && !twinData && !loading) {
      setLoading(true);
      runSimulation().finally(() => setLoading(false));
    }
  }, [patientData]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          Digital Twin Simulation
          <Button 
            variant="outline" 
            size="icon" 
            onClick={runSimulation} 
            disabled={simulating || !patientData}
          >
            {simulating ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          </Button>
        </CardTitle>
        <CardDescription>Virtual simulation of your gut microbiome ecosystem</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="bg-destructive/10 text-destructive p-3 rounded-md mb-4">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex flex-col items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary mb-2" />
            <p className="text-sm text-muted-foreground">Loading simulation data...</p>
          </div>
        ) : twinData ? (
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Inflammation Score</h3>
                <span className={`text-sm font-medium ${getScoreColor(twinData.inflammation_score, true)}`}>
                  {formatScore(twinData.inflammation_score, true)}
                </span>
              </div>
              <Progress value={getPercentageValue(twinData.inflammation_score, 0, 3, true)} 
                        className={getProgressColor(twinData.inflammation_score, true)} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Butyrate Flux</h3>
                <span className={`text-sm font-medium ${getScoreColor(twinData.butyrate_flux)}`}>
                  {formatScore(twinData.butyrate_flux)}
                </span>
              </div>
              <Progress value={getPercentageValue(twinData.butyrate_flux, 0, 1)} 
                        className={getProgressColor(twinData.butyrate_flux)} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Metabolic Health</h3>
                <span className={`text-sm font-medium ${getScoreColor(twinData.metabolic_health_score)}`}>
                  {formatScore(twinData.metabolic_health_score)}
                </span>
              </div>
              <Progress value={getPercentageValue(twinData.metabolic_health_score, 0, 100)} 
                        className={getProgressColor(twinData.metabolic_health_score)} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium">Gut Permeability</h3>
                <span className={`text-sm font-medium ${getScoreColor(twinData.gut_permeability_estimate, true)}`}>
                  {formatScore(twinData.gut_permeability_estimate, true)}
                </span>
              </div>
              <Progress value={getPercentageValue(twinData.gut_permeability_estimate, 0, 1, true)} 
                        className={getProgressColor(twinData.gut_permeability_estimate, true)} />
            </div>

            <div className="space-y-2 pt-4">
              <h3 className="text-sm font-medium">Recommendations</h3>
              <ul className="space-y-1">
                {twinData.recommendations.map((rec: string, i: number) => (
                  <li key={i} className="text-sm flex">
                    <span className="text-primary mr-2">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="space-y-2 pt-2">
              <h3 className="text-sm font-medium">Risk Factors</h3>
              <ul className="space-y-1">
                {twinData.risk_factors.map((risk: string, i: number) => (
                  <li key={i} className="text-sm flex">
                    <span className="text-destructive mr-2">•</span>
                    <span>{risk}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="text-xs text-muted-foreground mt-4 pt-2 border-t">
              Simulation confidence: {Math.round(twinData.simulation_confidence * 100)}%
            </div>
          </div>
        ) : patientData ? (
          <div className="flex flex-col items-center justify-center py-8">
            <Button onClick={runSimulation} disabled={simulating}>
              {simulating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Simulating...
                </>
              ) : (
                "Run Digital Twin Simulation"
              )}
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <p className="text-sm text-muted-foreground">
              Patient data is required to run the digital twin simulation.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Helper functions
function formatScore(value: number, isNegative = false): string {
  return isNegative 
    ? value.toFixed(1) + " (lower is better)"
    : value.toFixed(1) + " (higher is better)";
}

function getPercentageValue(value: number, min: number, max: number, isNegative = false): number {
  const percentage = ((value - min) / (max - min)) * 100;
  return isNegative ? 100 - percentage : percentage;
}

function getScoreColor(value: number, isNegative = false): string {
  if (isNegative) {
    // For values where lower is better
    if (value < 0.5) return "text-green-500";
    if (value < 1.0) return "text-emerald-600";
    if (value < 1.5) return "text-yellow-500";
    if (value < 2.0) return "text-orange-500";
    return "text-red-500";
  } else {
    // For values where higher is better
    if (value > 0.8) return "text-green-500";
    if (value > 0.6) return "text-emerald-600";
    if (value > 0.4) return "text-yellow-500";
    if (value > 0.2) return "text-orange-500";
    return "text-red-500";
  }
}

function getProgressColor(value: number, isNegative = false): string {
  if (isNegative) {
    // For values where lower is better
    if (value < 0.5) return "bg-green-500";
    if (value < 1.0) return "bg-emerald-600";
    if (value < 1.5) return "bg-yellow-500";
    if (value < 2.0) return "bg-orange-500";
    return "bg-red-500";
  } else {
    // For values where higher is better
    if (value > 0.8) return "bg-green-500";
    if (value > 0.6) return "bg-emerald-600";
    if (value > 0.4) return "bg-yellow-500";
    if (value > 0.2) return "bg-orange-500";
    return "bg-red-500";
  }
} 