"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import apiService, { TreatmentOption, TreatmentInput } from "@/lib/api-service";

interface TreatmentOptimizerProps {
  patientId: string;
  patientMetrics?: {
    inflammation: number;
    bloating: number;
    gut_permeability: number;
    butyrate_production: number;
    microbiome_diversity: number;
  };
}

// Example treatment options to choose from
const AVAILABLE_TREATMENTS: TreatmentOption[] = [
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
    name: "Low FODMAP Diet",
    type: "diet",
    estimated_effect: {
      inflammation: -0.1,
      bloating: -3,
      gut_permeability: -0.05,
      butyrate_production: -0.1,
      microbiome_diversity: -0.05
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
    name: "Probiotic (L. rhamnosus)",
    type: "supplement",
    estimated_effect: {
      inflammation: -0.15,
      bloating: -1.5,
      gut_permeability: -0.1,
      butyrate_production: 0.05,
      microbiome_diversity: 0.1
    }
  },
  {
    name: "Prebiotic (Inulin)",
    type: "supplement",
    estimated_effect: {
      inflammation: -0.1,
      bloating: 1,
      gut_permeability: -0.1,
      butyrate_production: 0.25,
      microbiome_diversity: 0.2
    }
  },
  {
    name: "Prebiotic (GOS)",
    type: "supplement",
    estimated_effect: {
      inflammation: -0.15,
      bloating: 0.5,
      gut_permeability: -0.1,
      butyrate_production: 0.2,
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
  },
  {
    name: "Digestive Enzymes",
    type: "supplement",
    estimated_effect: {
      inflammation: -0.05,
      bloating: -2,
      gut_permeability: 0,
      butyrate_production: 0,
      microbiome_diversity: 0
    }
  }
];

export default function TreatmentOptimizer({ patientId, patientMetrics }: TreatmentOptimizerProps) {
  const [loading, setLoading] = useState(false);
  const [treatmentCount, setTreatmentCount] = useState(3);
  const [selectedTreatments, setSelectedTreatments] = useState<TreatmentOption[]>([]);
  const [availableTreatments, setAvailableTreatments] = useState<TreatmentOption[]>(AVAILABLE_TREATMENTS);
  const [optimizationResult, setOptimizationResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const optimizeTreatment = async () => {
    if (!patientMetrics) {
      setError("Patient metrics not available");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Define current metrics from patient data
      const treatmentInput: TreatmentInput = {
        patient_id: patientId,
        current_metrics: { ...patientMetrics },
        available_treatments: availableTreatments,
        treatment_count: treatmentCount
      };

      const result = await apiService.optimizeTreatment(treatmentInput);
      setOptimizationResult(result);
      setSelectedTreatments(result.recommended_treatments);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to optimize treatment");
    } finally {
      setLoading(false);
    }
  };

  const formatChange = (value: number) => {
    const isPositive = value > 0;
    const prefix = isPositive ? '+' : '';
    return `${prefix}${value.toFixed(2)}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Treatment Optimizer</CardTitle>
        <CardDescription>Recommend optimal treatments based on patient metrics</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <div className="bg-destructive/10 text-destructive p-3 rounded-md mb-4">
            {error}
          </div>
        )}

        <div className="space-y-6">
          <div className="space-y-2">
            <Label>Number of treatments to recommend</Label>
            <div className="flex items-center gap-4">
              <Slider
                value={[treatmentCount]}
                min={1}
                max={5}
                step={1}
                onValueChange={(value) => setTreatmentCount(value[0])}
                className="flex-1"
              />
              <Input
                type="number"
                value={treatmentCount}
                onChange={(e) => setTreatmentCount(Number(e.target.value))}
                min={1}
                max={5}
                className="w-20"
              />
            </div>
          </div>

          {optimizationResult ? (
            <Tabs defaultValue="recommendations" className="mt-6">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
                <TabsTrigger value="analysis">Impact Analysis</TabsTrigger>
              </TabsList>
              <TabsContent value="recommendations" className="space-y-4 mt-4">
                <div className="space-y-4">
                  {selectedTreatments.map((treatment, index) => (
                    <div key={index} className="border rounded-md p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium">{treatment.name}</h3>
                        <Badge variant={treatment.type === "diet" ? "secondary" : "outline"}>
                          {treatment.type}
                        </Badge>
                      </div>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Metric</TableHead>
                            <TableHead>Expected Change</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(treatment.estimated_effect).map(([metric, value]) => (
                            <TableRow key={metric}>
                              <TableCell className="capitalize">{metric.replace('_', ' ')}</TableCell>
                              <TableCell className={value < 0 ? "text-green-600" : "text-amber-600"}>
                                {formatChange(value)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  ))}
                </div>
              </TabsContent>
              <TabsContent value="analysis" className="space-y-4 mt-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Combined Impact</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Metric</TableHead>
                          <TableHead>Current</TableHead>
                          <TableHead>Predicted</TableHead>
                          <TableHead>Change</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Object.entries(optimizationResult.predicted_improvement).map(([metric, change]) => (
                          <TableRow key={metric}>
                            <TableCell className="capitalize">{metric.replace('_', ' ')}</TableCell>
                            <TableCell>{patientMetrics?.[metric as keyof typeof patientMetrics]?.toFixed(2) || '-'}</TableCell>
                            <TableCell>
                              {((patientMetrics?.[metric as keyof typeof patientMetrics] || 0) - Number(change)).toFixed(2)}
                            </TableCell>
                            <TableCell className={Number(change) > 0 ? "text-green-600" : "text-amber-600"}>
                              {formatChange(-Number(change))}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    <div className="mt-4 text-sm text-muted-foreground">
                      Treatment confidence score: {Math.round(optimizationResult.confidence * 100)}%
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <Button 
              onClick={optimizeTreatment} 
              disabled={loading || !patientMetrics} 
              className="w-full mt-4"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Optimizing...
                </>
              ) : (
                "Optimize Treatment Plan"
              )}
            </Button>
          )}
        </div>
      </CardContent>
      <CardFooter className="flex justify-between border-t pt-4">
        {optimizationResult && (
          <Button variant="outline" onClick={() => setOptimizationResult(null)}>
            Reset
          </Button>
        )}
        {optimizationResult && (
          <Button>
            Apply to Patient Record
          </Button>
        )}
      </CardFooter>
    </Card>
  );
} 