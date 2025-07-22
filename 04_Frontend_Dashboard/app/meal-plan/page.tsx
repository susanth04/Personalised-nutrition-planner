"use client"

import { useState } from "react"
import { Header } from "@/components/header"
import { MealPlanForm } from "@/components/meal-plan-form"
import { MealPlanResults } from "@/components/meal-plan-results"
import { apiClient } from "@/lib/api-client"
import { useToast } from "@/hooks/use-toast"
import { Loader2 } from "lucide-react"

export default function MealPlanPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [mealPlanResults, setMealPlanResults] = useState(null)
  const { toast } = useToast()

  const handleSubmit = async (formData) => {
    setIsLoading(true)
    try {
      const response = await apiClient.post("/patient/meal-plan", formData)
      setMealPlanResults(response.data)
      toast({
        title: "Meal plan generated",
        description: "Your personalized meal plan is ready!",
      })
    } catch (error) {
      console.error("Error generating meal plan:", error)
      toast({
        variant: "destructive",
        title: "Failed to generate meal plan",
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
        <h1 className="mb-8 text-3xl font-bold text-center md:text-4xl">Personalized Meal Plan</h1>

        {isLoading ? (
          <div className="flex flex-col items-center justify-center p-8 space-y-4">
            <Loader2 className="w-12 h-12 text-emerald-600 animate-spin" />
            <p className="text-lg text-gray-600 dark:text-gray-300">Generating your personalized meal plan...</p>
          </div>
        ) : mealPlanResults ? (
          <MealPlanResults results={mealPlanResults} onReset={() => setMealPlanResults(null)} />
        ) : (
          <MealPlanForm onSubmit={handleSubmit} />
        )}
      </div>
    </main>
  )
}
