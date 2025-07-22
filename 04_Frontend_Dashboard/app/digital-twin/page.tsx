"use client"

import { useState } from "react"
import { Header } from "@/components/header"
import { DigitalTwinForm } from "@/components/digital-twin-form"
import { DigitalTwinResults } from "@/components/digital-twin-results"
import { apiClient } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"
import { Loader2 } from "lucide-react"

export default function DigitalTwinPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [simulationResults, setSimulationResults] = useState(null)
  const { toast } = useToast()

  const handleSubmit = async (formData) => {
    setIsLoading(true)
    try {
      const response = await apiClient.post("/digital-twin/simulate", formData)
      setSimulationResults(response.data)
      toast({
        title: "Simulation complete",
        description: "Your digital twin simulation results are ready!",
      })
    } catch (error) {
      console.error("Error running simulation:", error)
      toast({
        variant: "destructive",
        title: "Failed to run simulation",
        description: error.response?.data?.detail || "Please try again later.",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-gray-50 dark:from-gray-950 dark:to-gray-900">
      <Header />
      <div className="container px-4 py-8 mx-auto max-w-7xl">
        <h1 className="mb-8 text-3xl font-bold text-center md:text-4xl">Digital Twin Simulation</h1>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center p-8 space-y-4">
            <Loader2 className="w-12 h-12 text-emerald-600 animate-spin" />
            <p className="text-lg text-gray-600 dark:text-gray-300">Running your digital twin simulation...</p>
          </div>
        ) : simulationResults ? (
          <DigitalTwinResults results={simulationResults} onReset={() => setSimulationResults(null)} />
        ) : (
          <DigitalTwinForm onSubmit={handleSubmit} />
        )}
      </div>
    </main>
  )
}
