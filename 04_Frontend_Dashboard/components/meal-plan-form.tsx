"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Plus, Trash2 } from "lucide-react"

export function MealPlanForm({ onSubmit }) {
  const [formData, setFormData] = useState({
    age: 35,
    weight: 70,
    height: 170,
    diet_type: "Omnivore",
    calories_target: 2000,
    symptoms: {
      bloating: 3,
      abdominal_pain: 2,
      diarrhea: 1,
      constipation: 2,
    },
    recent_foods: [{ food_name: "", fiber_g: 0, prebiotic_score: 5 }],
  })

  const [errors, setErrors] = useState({})

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }))
  }

  const handleSymptomChange = (symptom, value) => {
    setFormData((prev) => ({
      ...prev,
      symptoms: {
        ...prev.symptoms,
        [symptom]: value[0],
      },
    }))
  }

  const handleDietTypeChange = (value) => {
    setFormData((prev) => ({
      ...prev,
      diet_type: value,
    }))
  }

  const handleFoodChange = (index, field, value) => {
    const updatedFoods = [...formData.recent_foods]
    updatedFoods[index] = {
      ...updatedFoods[index],
      [field]: value,
    }
    setFormData((prev) => ({
      ...prev,
      recent_foods: updatedFoods,
    }))
  }

  const addFood = () => {
    setFormData((prev) => ({
      ...prev,
      recent_foods: [...prev.recent_foods, { food_name: "", fiber_g: 0, prebiotic_score: 5 }],
    }))
  }

  const removeFood = (index) => {
    if (formData.recent_foods.length > 1) {
      const updatedFoods = [...formData.recent_foods]
      updatedFoods.splice(index, 1)
      setFormData((prev) => ({
        ...prev,
        recent_foods: updatedFoods,
      }))
    }
  }

  const validateForm = () => {
    const newErrors = {}

    if (formData.age < 18 || formData.age > 100) {
      newErrors.age = "Age must be between 18 and 100"
    }

    if (formData.weight < 30 || formData.weight > 200) {
      newErrors.weight = "Weight must be between 30 and 200 kg"
    }

    if (formData.height < 100 || formData.height > 250) {
      newErrors.height = "Height must be between 100 and 250 cm"
    }

    if (formData.calories_target < 1000 || formData.calories_target > 4000) {
      newErrors.calories_target = "Calories must be between 1000 and 4000"
    }

    formData.recent_foods.forEach((food, index) => {
      if (!food.food_name.trim()) {
        newErrors[`food_name_${index}`] = "Food name is required"
      }

      if (food.fiber_g < 0 || food.fiber_g > 100) {
        newErrors[`fiber_g_${index}`] = "Fiber must be between 0 and 100g"
      }
    })

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (validateForm()) {
      onSubmit(formData)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Personal Information</CardTitle>
          <CardDescription>Enter your basic health information to personalize your meal plan.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="age">Age</Label>
            <Input
              id="age"
              name="age"
              type="number"
              min="18"
              max="100"
              value={String(formData.age)}
              onChange={handleChange}
              className={errors.age ? "border-red-500" : ""}
            />
            {errors.age && <p className="text-sm text-red-500">{errors.age}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="weight">Weight (kg)</Label>
            <Input
              id="weight"
              name="weight"
              type="number"
              min="30"
              max="200"
              step="0.1"
              value={String(formData.weight)}
              onChange={handleChange}
              className={errors.weight ? "border-red-500" : ""}
            />
            {errors.weight && <p className="text-sm text-red-500">{errors.weight}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="height">Height (cm)</Label>
            <Input
              id="height"
              name="height"
              type="number"
              min="100"
              max="250"
              value={String(formData.height)}
              onChange={handleChange}
              className={errors.height ? "border-red-500" : ""}
            />
            {errors.height && <p className="text-sm text-red-500">{errors.height}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="diet-type">Diet Type</Label>
            <Select value={formData.diet_type} onValueChange={handleDietTypeChange}>
              <SelectTrigger id="diet-type">
                <SelectValue placeholder="Select diet type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Omnivore">Omnivore</SelectItem>
                <SelectItem value="Vegetarian">Vegetarian</SelectItem>
                <SelectItem value="Vegan">Vegan</SelectItem>
                <SelectItem value="Low FODMAP">Low FODMAP</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="calories_target">Daily Calories Target</Label>
            <Input
              id="calories_target"
              name="calories_target"
              type="number"
              min="1000"
              max="4000"
              step="50"
              value={String(formData.calories_target)}
              onChange={handleChange}
              className={errors.calories_target ? "border-red-500" : ""}
            />
            {errors.calories_target && <p className="text-sm text-red-500">{errors.calories_target}</p>}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Symptom Assessment</CardTitle>
          <CardDescription>Rate your digestive symptoms from 0 (none) to 10 (severe).</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-8 md:grid-cols-2">
          <div className="space-y-4">
            <Label>Bloating: {formData.symptoms.bloating}</Label>
            <Slider
              value={[formData.symptoms.bloating]}
              min={0}
              max={10}
              step={1}
              onValueChange={(value) => handleSymptomChange("bloating", value)}
            />
          </div>

          <div className="space-y-4">
            <Label>Abdominal Pain: {formData.symptoms.abdominal_pain}</Label>
            <Slider
              value={[formData.symptoms.abdominal_pain]}
              min={0}
              max={10}
              step={1}
              onValueChange={(value) => handleSymptomChange("abdominal_pain", value)}
            />
          </div>

          <div className="space-y-4">
            <Label>Diarrhea: {formData.symptoms.diarrhea}</Label>
            <Slider
              value={[formData.symptoms.diarrhea]}
              min={0}
              max={10}
              step={1}
              onValueChange={(value) => handleSymptomChange("diarrhea", value)}
            />
          </div>

          <div className="space-y-4">
            <Label>Constipation: {formData.symptoms.constipation}</Label>
            <Slider
              value={[formData.symptoms.constipation]}
              min={0}
              max={10}
              step={1}
              onValueChange={(value) => handleSymptomChange("constipation", value)}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent Food Intake</CardTitle>
          <CardDescription>Add foods you've consumed recently to help personalize your meal plan.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {formData.recent_foods.map((food, index) => (
            <div key={index} className="p-4 border rounded-lg">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-medium">Food Item {index + 1}</h4>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => removeFood(index)}
                  disabled={formData.recent_foods.length === 1}
                >
                  <Trash2 className="w-4 h-4" />
                  <span className="sr-only">Remove food</span>
                </Button>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor={`food-name-${index}`}>Food Name</Label>
                  <Input
                    id={`food-name-${index}`}
                    value={food.food_name}
                    onChange={(e) => handleFoodChange(index, "food_name", e.target.value)}
                    className={errors[`food_name_${index}`] ? "border-red-500" : ""}
                  />
                  {errors[`food_name_${index}`] && (
                    <p className="text-sm text-red-500">{errors[`food_name_${index}`]}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor={`fiber-${index}`}>Fiber (g)</Label>
                  <Input
                    id={`fiber-${index}`}
                    name={`fiber-${index}`}
                    type="number"
                    min="0"
                    max="100"
                    step="0.1"
                    value={String(food.fiber_g)}
                    onChange={(e) => handleFoodChange(index, "fiber_g", Number(e.target.value))}
                    className={errors[`fiber_g_${index}`] ? "border-red-500" : ""}
                  />
                  {errors[`fiber_g_${index}`] && <p className="text-sm text-red-500">{errors[`fiber_g_${index}`]}</p>}
                </div>

                <div className="space-y-2">
                  <Label htmlFor={`prebiotic-${index}`}>Prebiotic Score: {food.prebiotic_score}</Label>
                  <Slider
                    id={`prebiotic-${index}`}
                    value={[food.prebiotic_score]}
                    min={0}
                    max={10}
                    step={1}
                    onValueChange={(value) => handleFoodChange(index, "prebiotic_score", value[0])}
                  />
                </div>
              </div>
            </div>
          ))}

          <Button type="button" variant="outline" onClick={addFood} className="w-full">
            <Plus className="w-4 h-4 mr-2" />
            Add Another Food
          </Button>
        </CardContent>
        <CardFooter>
          <Button type="submit" className="w-full">
            Generate Meal Plan
          </Button>
        </CardFooter>
      </Card>
    </form>
  )
}
