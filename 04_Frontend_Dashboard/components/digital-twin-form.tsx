"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"

export function DigitalTwinForm({ onSubmit }) {
  const [formData, setFormData] = useState({
    patient_id: `patient-${Math.floor(Math.random() * 10000)}`,
    age: 35,
    weight: 70,
    height: 170,
    daily_fiber: 25,
    calories_intake: 2000,
    symptoms: {
      bloating: 3,
      abdominal_pain: 2,
      diarrhea: 1,
      constipation: 2,
    },
    microbiome_diversity: 0.6,
    calprotectin: null,
    additional_features: {},
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

    if (formData.daily_fiber < 0 || formData.daily_fiber > 100) {
      newErrors.daily_fiber = "Fiber must be between 0 and 100g"
    }

    if (formData.calories_intake < 1000 || formData.calories_intake > 4000) {
      newErrors.calories_intake = "Calories must be between 1000 and 4000"
    }

    if (
      formData.microbiome_diversity !== null &&
      (formData.microbiome_diversity < 0 || formData.microbiome_diversity > 1)
    ) {
      newErrors.microbiome_diversity = "Diversity must be between 0 and 1"
    }

    if (formData.calprotectin !== null && formData.calprotectin < 0) {
      newErrors.calprotectin = "Calprotectin must be a positive number"
    }

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
          <CardDescription>Enter your basic health information for the digital twin simulation.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <Label htmlFor="patient_id">Patient ID</Label>
            <Input
              id="patient_id"
              name="patient_id"
              value={formData.patient_id}
              onChange={handleChange}
              className={errors.patient_id ? "border-red-500" : ""}
            />
            {errors.patient_id && <p className="text-sm text-red-500">{errors.patient_id}</p>}
          </div>

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
            <Label htmlFor="daily_fiber">Daily Fiber Intake (g)</Label>
            <Input
              id="daily_fiber"
              name="daily_fiber"
              type="number"
              min="0"
              max="100"
              step="0.5"
              value={String(formData.daily_fiber)}
              onChange={handleChange}
              className={errors.daily_fiber ? "border-red-500" : ""}
            />
            {errors.daily_fiber && <p className="text-sm text-red-500">{errors.daily_fiber}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="calories_intake">Daily Calories Intake</Label>
            <Input
              id="calories_intake"
              name="calories_intake"
              type="number"
              min="1000"
              max="4000"
              step="50"
              value={String(formData.calories_intake)}
              onChange={handleChange}
              className={errors.calories_intake ? "border-red-500" : ""}
            />
            {errors.calories_intake && <p className="text-sm text-red-500">{errors.calories_intake}</p>}
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
          <CardTitle>Advanced Metrics (Optional)</CardTitle>
          <CardDescription>Add additional health metrics if available for more accurate simulation.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-6 md:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="microbiome_diversity">Microbiome Diversity (0-1)</Label>
            <Input
              id="microbiome_diversity"
              name="microbiome_diversity"
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={String(formData.microbiome_diversity)}
              onChange={handleChange}
              className={errors.microbiome_diversity ? "border-red-500" : ""}
            />
            {errors.microbiome_diversity && <p className="text-sm text-red-500">{errors.microbiome_diversity}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="calprotectin">Calprotectin (µg/g, if available)</Label>
            <Input
              id="calprotectin"
              name="calprotectin"
              type="number"
              min="0"
              step="1"
              placeholder="Optional"
              value={formData.calprotectin === null ? "" : String(formData.calprotectin)}
              onChange={(e) => {
                const value = e.target.value === "" ? null : Number(e.target.value)
                setFormData((prev) => ({
                  ...prev,
                  calprotectin: value,
                }))
              }}
              className={errors.calprotectin ? "border-red-500" : ""}
            />
            {errors.calprotectin && <p className="text-sm text-red-500">{errors.calprotectin}</p>}
          </div>
        </CardContent>
        <CardFooter>
          <Button type="submit" className="w-full">
            Run Digital Twin Simulation
          </Button>
        </CardFooter>
      </Card>
    </form>
  )
}
