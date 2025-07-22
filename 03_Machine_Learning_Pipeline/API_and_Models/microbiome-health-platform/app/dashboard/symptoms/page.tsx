"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-provider"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { useToast } from "@/hooks/use-toast"
import { Checkbox } from "@/components/ui/checkbox"

export default function SymptomsPage() {
  const [symptoms, setSymptoms] = useState({
    bloating: false,
    constipation: false,
    diarrhea: false,
    abdominalPain: false,
    fatigue: false,
    headache: false,
    jointPain: false,
    skinIssues: false,
    other: false,
  })
  const [severity, setSeverity] = useState(5)
  const [notes, setNotes] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { user } = useAuth()
  const router = useRouter()
  const { toast } = useToast()
  const supabase = createClient()

  const handleSymptomChange = (symptom: keyof typeof symptoms) => {
    setSymptoms((prev) => ({
      ...prev,
      [symptom]: !prev[symptom],
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!user) {
      toast({
        title: "Error",
        description: "You must be logged in to submit symptoms",
        variant: "destructive",
      })
      return
    }

    // Check if at least one symptom is selected
    const hasSymptoms = Object.values(symptoms).some((value) => value)
    if (!hasSymptoms) {
      toast({
        title: "Error",
        description: "Please select at least one symptom",
        variant: "destructive",
      })
      return
    }

    setIsSubmitting(true)

    try {
      // Save symptoms to database
      const { error } = await supabase.from("symptoms").insert([
        {
          user_id: user.id,
          symptoms: Object.entries(symptoms)
            .filter(([_, value]) => value)
            .map(([key, _]) => key),
          severity,
          notes,
          date: new Date().toISOString(),
        },
      ])

      if (error) {
        throw error
      }

      toast({
        title: "Success",
        description: "Symptoms logged successfully",
      })

      // Reset form
      setSymptoms({
        bloating: false,
        constipation: false,
        diarrhea: false,
        abdominalPain: false,
        fatigue: false,
        headache: false,
        jointPain: false,
        skinIssues: false,
        other: false,
      })
      setSeverity(5)
      setNotes("")

      // Redirect to dashboard
      router.push("/dashboard")
      router.refresh()
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "An error occurred while logging symptoms",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Track Symptoms</h1>
        <p className="text-muted-foreground">
          Log your symptoms to help identify patterns related to your microbiome health
        </p>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Log Symptoms</CardTitle>
            <CardDescription>Select the symptoms you're experiencing today</CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <Label>Symptoms</Label>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(symptoms).map(([key, value]) => (
                    <div key={key} className="flex items-center space-x-2">
                      <Checkbox
                        id={key}
                        checked={value}
                        onCheckedChange={() => handleSymptomChange(key as keyof typeof symptoms)}
                      />
                      <Label htmlFor={key} className="capitalize">
                        {key === "abdominalPain"
                          ? "Abdominal Pain"
                          : key === "skinIssues"
                            ? "Skin Issues"
                            : key === "jointPain"
                              ? "Joint Pain"
                              : key}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="severity">Severity (1-10)</Label>
                  <span className="text-sm font-medium">{severity}</span>
                </div>
                <Slider
                  id="severity"
                  min={1}
                  max={10}
                  step={1}
                  value={[severity]}
                  onValueChange={(value) => setSeverity(value[0])}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Mild</span>
                  <span>Moderate</span>
                  <span>Severe</span>
                </div>
              </div>
              <div className="space-y-2">
                <Label htmlFor="notes">Additional Notes</Label>
                <Textarea
                  id="notes"
                  placeholder="Enter any additional details about your symptoms"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </div>
            </CardContent>
            <CardFooter>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Submitting..." : "Submit"}
              </Button>
            </CardFooter>
          </form>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Symptom History</CardTitle>
            <CardDescription>Your recently logged symptoms</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              {[
                { date: "2023-04-08", symptoms: ["Bloating", "Fatigue"], severity: 6 },
                { date: "2023-04-05", symptoms: ["Headache", "Joint Pain"], severity: 4 },
                { date: "2023-04-01", symptoms: ["Constipation", "Abdominal Pain"], severity: 7 },
              ].map((entry, i) => (
                <div key={i} className="border rounded-md p-3 space-y-2">
                  <div className="flex justify-between">
                    <p className="font-medium">{new Date(entry.date).toLocaleDateString()}</p>
                    <p className="text-sm">Severity: {entry.severity}/10</p>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {entry.symptoms.map((symptom, j) => (
                      <span key={j} className="bg-muted text-xs px-2 py-1 rounded-md">
                        {symptom}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
