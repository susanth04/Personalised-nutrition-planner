"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, AlertCircle, CheckCircle, Info } from "lucide-react"

export function DigitalTwinResults({ results, onReset }) {
  if (!results) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <p>No simulation results available. Please run a simulation first.</p>
          <Button onClick={onReset} className="mt-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Form
          </Button>
        </CardContent>
      </Card>
    )
  }

  // Helper function to determine risk level color
  const getRiskColor = (value, metric) => {
    if (metric === "metabolic_health_score") {
      // Higher is better for metabolic health
      if (value >= 80) return "text-green-600 dark:text-green-500"
      if (value >= 60) return "text-yellow-600 dark:text-yellow-500"
      return "text-red-600 dark:text-red-500"
    } else if (metric === "gut_permeability_estimate") {
      // Lower is better for gut permeability
      if (value <= 0.3) return "text-green-600 dark:text-green-500"
      if (value <= 0.6) return "text-yellow-600 dark:text-yellow-500"
      return "text-red-600 dark:text-red-500"
    } else if (metric === "inflammation_score") {
      // Lower is better for inflammation
      if (value <= 1.0) return "text-green-600 dark:text-green-500"
      if (value <= 2.0) return "text-yellow-600 dark:text-yellow-500"
      return "text-red-600 dark:text-red-500"
    } else {
      // Default case (butyrate_flux) - higher is better
      if (value >= 0.5) return "text-green-600 dark:text-green-500"
      if (value >= 0.3) return "text-yellow-600 dark:text-yellow-500"
      return "text-red-600 dark:text-red-500"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Digital Twin Simulation Results</h2>
        <Button variant="outline" onClick={onReset}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Form
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Health Metrics</CardTitle>
            <CardDescription>Key health indicators from your digital twin simulation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Butyrate Flux</span>
                <span className={`text-sm font-medium ${getRiskColor(results.butyrate_flux, "butyrate_flux")}`}>
                  {results.butyrate_flux.toFixed(2)}
                </span>
              </div>
              <Progress value={(results.butyrate_flux / 1) * 100} className="h-2" />
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Measures the production rate of butyrate in your gut. Higher is better.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Inflammation Score</span>
                <span
                  className={`text-sm font-medium ${getRiskColor(results.inflammation_score, "inflammation_score")}`}
                >
                  {results.inflammation_score.toFixed(2)}
                </span>
              </div>
              <Progress value={(results.inflammation_score / 3) * 100} className="h-2" />
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Indicates level of gut inflammation. Lower is better.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Metabolic Health Score</span>
                <span
                  className={`text-sm font-medium ${getRiskColor(results.metabolic_health_score, "metabolic_health_score")}`}
                >
                  {results.metabolic_health_score.toFixed(1)}
                </span>
              </div>
              <Progress value={results.metabolic_health_score} className="h-2" />
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Overall metabolic health rating (0-100). Higher is better.
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Gut Permeability</span>
                <span
                  className={`text-sm font-medium ${getRiskColor(results.gut_permeability_estimate, "gut_permeability_estimate")}`}
                >
                  {results.gut_permeability_estimate.toFixed(2)}
                </span>
              </div>
              <Progress value={results.gut_permeability_estimate * 100} className="h-2" />
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Estimated gut barrier function (0-1). Lower is better.
              </p>
            </div>

            <div className="flex items-center justify-between p-3 mt-4 bg-gray-50 rounded-md dark:bg-gray-800">
              <div className="flex items-center">
                <Info className="w-4 h-4 mr-2 text-blue-500" />
                <span className="text-sm">Simulation Confidence</span>
              </div>
              <span className="text-sm font-medium">{(results.simulation_confidence * 100).toFixed(0)}%</span>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recommendations</CardTitle>
              <CardDescription>Personalized health suggestions based on your results</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {results.recommendations.map((recommendation, index) => (
                  <li key={index} className="flex items-start">
                    <CheckCircle className="w-5 h-5 mr-2 text-emerald-600 shrink-0 dark:text-emerald-500" />
                    <span>{recommendation}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Risk Factors</CardTitle>
              <CardDescription>Health concerns identified in your simulation</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {results.risk_factors.map((risk, index) => (
                  <li key={index} className="flex items-start">
                    <AlertCircle className="w-5 h-5 mr-2 text-orange-600 shrink-0 dark:text-orange-500" />
                    <span>{risk}</span>
                  </li>
                ))}
                {results.risk_factors.length === 0 && (
                  <li className="text-gray-500 dark:text-gray-400">
                    No significant risk factors identified at this time.
                  </li>
                )}
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
