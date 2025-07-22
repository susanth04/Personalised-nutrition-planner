import { Header } from "@/components/header"
import { ApiStatus } from "@/components/api-status"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ArrowRight, Utensils, Activity } from "lucide-react"

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-gray-50 dark:from-gray-950 dark:to-gray-900">
      <Header />
      <div className="container px-4 py-12 mx-auto max-w-7xl">
        <div className="flex flex-col items-center justify-center space-y-8 text-center">
          <ApiStatus />

          <h1 className="text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl">
            Personalized <span className="text-emerald-600 dark:text-emerald-500">Nutrition</span> Dashboard
          </h1>

          <p className="max-w-2xl text-xl text-gray-600 dark:text-gray-300">
            Get personalized meal plans and health insights based on your unique health profile, symptoms, and dietary
            preferences.
          </p>

          <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4">
            <Button asChild size="lg" className="gap-2">
              <Link href="/meal-plan">
                <Utensils className="w-5 h-5" />
                Create Meal Plan
                <ArrowRight className="w-4 h-4 ml-1" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg" className="gap-2">
              <Link href="/digital-twin">
                <Activity className="w-5 h-5" />
                Digital Twin Simulation
                <ArrowRight className="w-4 h-4 ml-1" />
              </Link>
            </Button>
          </div>
        </div>

        <div className="grid gap-8 mt-16 md:grid-cols-2 lg:grid-cols-3">
          <div className="p-6 bg-white rounded-lg shadow-md dark:bg-gray-800">
            <h2 className="mb-3 text-xl font-bold">Personalized Meal Plans</h2>
            <p className="text-gray-600 dark:text-gray-300">
              Get meal recommendations tailored to your health needs, dietary preferences, and symptom profile.
            </p>
          </div>

          <div className="p-6 bg-white rounded-lg shadow-md dark:bg-gray-800">
            <h2 className="mb-3 text-xl font-bold">Digital Twin Simulation</h2>
            <p className="text-gray-600 dark:text-gray-300">
              Simulate your gut health metrics and receive personalized recommendations based on your health data.
            </p>
          </div>

          <div className="p-6 bg-white rounded-lg shadow-md dark:bg-gray-800">
            <h2 className="mb-3 text-xl font-bold">Advanced AI Analysis</h2>
            <p className="text-gray-600 dark:text-gray-300">
              Our system uses XGBoost machine learning to predict health outcomes and optimize nutrition plans.
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
