"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, Utensils, Apple, Coffee, ChefHat } from "lucide-react"

export function MealPlanResults({ results, onReset }) {
  const [activeDay, setActiveDay] = useState("day-1")

  if (!results || !results.meal_plan) {
    return (
      <Card>
        <CardContent className="p-6 text-center">
          <p>No meal plan data available. Please generate a meal plan first.</p>
          <Button onClick={onReset} className="mt-4">
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Form
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Your Personalized Meal Plan</h2>
        <Button variant="outline" onClick={onReset}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Form
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Nutrition Summary</CardTitle>
          <CardDescription>Key metrics from your personalized nutrition plan</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 md:grid-cols-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Fiber Target</span>
              <span className="text-sm font-medium">{results.fiber_target}g</span>
            </div>
            <Progress value={(results.fiber_target / 40) * 100} className="h-2" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Butyrate Score</span>
              <span className="text-sm font-medium">{results.butyrate_score.toFixed(2)}</span>
            </div>
            <Progress value={(results.butyrate_score / 1) * 100} className="h-2 bg-emerald-100 dark:bg-emerald-950" />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Inflammation Score</span>
              <span className="text-sm font-medium">{results.inflammation_score.toFixed(2)}</span>
            </div>
            <Progress value={(results.inflammation_score / 3) * 100} className="h-2 bg-orange-100 dark:bg-orange-950" />
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="day-1" value={activeDay} onValueChange={setActiveDay}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="day-1">Day 1</TabsTrigger>
          <TabsTrigger value="day-2">Day 2</TabsTrigger>
          <TabsTrigger value="day-3">Day 3</TabsTrigger>
        </TabsList>

        {results.meal_plan.map((dayPlan, dayIndex) => (
          <TabsContent key={`day-${dayIndex + 1}`} value={`day-${dayIndex + 1}`}>
            <Card>
              <CardHeader>
                <CardTitle>Day {dayIndex + 1} Meal Plan</CardTitle>
                <CardDescription>
                  Total: {dayPlan.nutrition.calories} calories, {dayPlan.nutrition.protein}g protein,
                  {dayPlan.nutrition.fiber}g fiber, {dayPlan.nutrition.fat}g fat
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Accordion type="single" collapsible className="w-full">
                  <AccordionItem value="breakfast">
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center">
                        <Coffee className="w-5 h-5 mr-2 text-emerald-600 dark:text-emerald-500" />
                        <span>Breakfast</span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <MealCard meal={dayPlan.breakfast} />
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="lunch">
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center">
                        <Utensils className="w-5 h-5 mr-2 text-emerald-600 dark:text-emerald-500" />
                        <span>Lunch</span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <MealCard meal={dayPlan.lunch} />
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem value="dinner">
                    <AccordionTrigger className="hover:no-underline">
                      <div className="flex items-center">
                        <ChefHat className="w-5 h-5 mr-2 text-emerald-600 dark:text-emerald-500" />
                        <span>Dinner</span>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <MealCard meal={dayPlan.dinner} />
                    </AccordionContent>
                  </AccordionItem>

                  {dayPlan.snack && (
                    <AccordionItem value="snack">
                      <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-center">
                          <Apple className="w-5 h-5 mr-2 text-emerald-600 dark:text-emerald-500" />
                          <span>Snack</span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <MealCard meal={dayPlan.snack} />
                      </AccordionContent>
                    </AccordionItem>
                  )}
                </Accordion>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  )
}

function MealCard({ meal }) {
  return (
    <div className="p-4 border rounded-lg">
      <div className="flex flex-col space-y-4">
        <div>
          <h3 className="text-lg font-semibold">{meal.name}</h3>
          <p className="text-sm text-gray-600 dark:text-gray-300">{meal.description}</p>
        </div>

        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          <div className="p-2 bg-gray-50 rounded-md dark:bg-gray-800">
            <p className="text-xs text-gray-500 dark:text-gray-400">Calories</p>
            <p className="font-medium">{meal.nutrition.calories}</p>
          </div>
          <div className="p-2 bg-gray-50 rounded-md dark:bg-gray-800">
            <p className="text-xs text-gray-500 dark:text-gray-400">Protein</p>
            <p className="font-medium">{meal.nutrition.protein}g</p>
          </div>
          <div className="p-2 bg-gray-50 rounded-md dark:bg-gray-800">
            <p className="text-xs text-gray-500 dark:text-gray-400">Fiber</p>
            <p className="font-medium">{meal.nutrition.fiber}g</p>
          </div>
          <div className="p-2 bg-gray-50 rounded-md dark:bg-gray-800">
            <p className="text-xs text-gray-500 dark:text-gray-400">Fat</p>
            <p className="font-medium">{meal.nutrition.fat}g</p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {meal.tags.map((tag, index) => (
            <Badge key={index} variant="outline" className="capitalize">
              {tag.replace(/_/g, " ")}
            </Badge>
          ))}
          <Badge className="bg-emerald-100 text-emerald-800 hover:bg-emerald-200 dark:bg-emerald-900 dark:text-emerald-300 dark:hover:bg-emerald-800">
            Butyrate: {meal.butyrate_potential.toFixed(1)}/10
          </Badge>
        </div>
      </div>
    </div>
  )
}
