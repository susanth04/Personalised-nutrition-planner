"use client"

import { useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Legend, BarChart, Bar } from "recharts"
import { ArrowLeft, FileUp, ClipboardList, BarChart3, MessageSquare } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export default function PatientDetailPage() {
  const params = useParams()
  const patientId = params.id as string
  const router = useRouter()
  const { toast } = useToast()
  const [recommendation, setRecommendation] = useState("")
  const [isSending, setIsSending] = useState(false)

  // Sample patient data - in a real app, this would be fetched from the database
  const patient = {
    id: patientId,
    name: "John Doe",
    email: "john@example.com",
    age: 35,
    gender: "Male",
    healthScore: 78,
    lastUpload: "2023-04-08",
    uploads: 3,
    symptoms: 12,
  }

  // Sample data for charts
  const healthData = [
    { date: "Jan", score: 65 },
    { date: "Feb", score: 68 },
    { date: "Mar", score: 75 },
    { date: "Apr", score: 72 },
    { date: "May", score: 78 },
    { date: "Jun", score: 82 },
  ]

  const symptomData = [
    { name: "Bloating", count: 8 },
    { name: "Fatigue", count: 6 },
    { name: "Headache", count: 4 },
    { name: "Joint Pain", count: 3 },
    { name: "Skin Issues", count: 2 },
  ]

  const handleSendRecommendation = async () => {
    if (!recommendation.trim()) {
      toast({
        title: "Error",
        description: "Please enter a recommendation",
        variant: "destructive",
      })
      return
    }

    setIsSending(true)

    try {
      // In a real app, this would send the recommendation to the backend
      await new Promise((resolve) => setTimeout(resolve, 1000))

      toast({
        title: "Success",
        description: "Recommendation sent to patient",
      })

      setRecommendation("")
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to send recommendation",
        variant: "destructive",
      })
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={() => router.back()}>
          <ArrowLeft className="h-4 w-4" />
          <span className="sr-only">Back</span>
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{patient.name}</h1>
          <p className="text-muted-foreground">Patient ID: {patient.id}</p>
        </div>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Health Score</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{patient.healthScore}/100</div>
            <p className="text-xs text-muted-foreground">+5 from last month</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Uploads</CardTitle>
            <FileUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{patient.uploads}</div>
            <p className="text-xs text-muted-foreground">
              Last upload: {new Date(patient.lastUpload).toLocaleDateString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Symptom Logs</CardTitle>
            <ClipboardList className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{patient.symptoms}</div>
            <p className="text-xs text-muted-foreground">+4 from last month</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Demographics</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-md font-medium">
              {patient.age} years, {patient.gender}
            </div>
            <p className="text-xs text-muted-foreground">{patient.email}</p>
          </CardContent>
        </Card>
      </div>
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="microbiome">Microbiome Data</TabsTrigger>
          <TabsTrigger value="symptoms">Symptoms</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Health Score Trend</CardTitle>
              <CardDescription>Patient's health score over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ChartContainer
                  config={{
                    score: {
                      label: "Health Score",
                      color: "hsl(var(--chart-1))",
                    },
                  }}
                  className="h-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={healthData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis domain={[0, 100]} />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend />
                      <Line type="monotone" dataKey="score" stroke="var(--color-score)" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
            </CardContent>
          </Card>
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Recent Activity</CardTitle>
                <CardDescription>Patient's recent actions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { action: "Uploaded microbiome data", date: "2 days ago" },
                    { action: "Logged new symptoms", date: "3 days ago" },
                    { action: "Viewed health insights", date: "1 week ago" },
                    { action: "Updated profile information", date: "2 weeks ago" },
                  ].map((activity, i) => (
                    <div key={i} className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-primary mr-2"></div>
                      <div className="flex-1 space-y-1">
                        <p className="text-sm font-medium leading-none">{activity.action}</p>
                        <p className="text-sm text-muted-foreground">{activity.date}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Patient Notes</CardTitle>
                <CardDescription>Clinical notes and observations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between">
                      <p className="font-medium">Initial Assessment</p>
                      <p className="text-sm text-muted-foreground">2023-03-15</p>
                    </div>
                    <p className="text-sm">
                      Patient reports occasional bloating and fatigue. Microbiome analysis shows good diversity but
                      slightly elevated Proteobacteria.
                    </p>
                  </div>
                  <div className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between">
                      <p className="font-medium">Follow-up</p>
                      <p className="text-sm text-muted-foreground">2023-04-01</p>
                    </div>
                    <p className="text-sm">
                      Dietary changes have improved symptoms. Health score increased by 5 points. Continue monitoring.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="microbiome" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Microbiome Data</CardTitle>
              <CardDescription>Patient's microbiome analysis results</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Key Findings</h3>
                <ul className="list-disc pl-5 text-sm space-y-1">
                  <li>Healthy levels of Bifidobacterium and Lactobacillus species</li>
                  <li>Slightly elevated Proteobacteria, which may indicate mild inflammation</li>
                  <li>Good diversity of beneficial bacteria species</li>
                  <li>Low levels of potentially harmful bacteria</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Comparison to Population</h3>
                <p className="text-sm">
                  Patient's microbiome diversity is in the 65th percentile compared to others in their age group, which
                  is good. The Firmicutes to Bacteroidetes ratio is within the healthy range.
                </p>
              </div>
              <div className="h-[300px] bg-muted rounded-md flex items-center justify-center">
                <p className="text-muted-foreground">Microbiome Composition Chart</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="symptoms" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Symptom History</CardTitle>
              <CardDescription>Patient's reported symptoms</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-[300px]">
                <ChartContainer
                  config={{
                    count: {
                      label: "Occurrence Count",
                      color: "hsl(var(--chart-1))",
                    },
                  }}
                  className="h-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={symptomData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend />
                      <Bar dataKey="count" fill="var(--color-count)" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Recent Symptoms</h3>
                {[
                  { symptom: "Bloating", severity: 6, date: "2023-04-08", notes: "After eating dairy products" },
                  { symptom: "Fatigue", severity: 4, date: "2023-04-05", notes: "Mid-afternoon, after lunch" },
                  { symptom: "Headache", severity: 5, date: "2023-04-01", notes: "Morning, before breakfast" },
                ].map((entry, i) => (
                  <div key={i} className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between">
                      <p className="font-medium">{entry.symptom}</p>
                      <p className="text-sm">Severity: {entry.severity}/10</p>
                    </div>
                    <p className="text-sm text-muted-foreground">{new Date(entry.date).toLocaleDateString()}</p>
                    <p className="text-sm">{entry.notes}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="recommendations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Previous Recommendations</CardTitle>
              <CardDescription>Health recommendations provided to the patient</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    date: "2023-04-01",
                    recommendation:
                      "Increase fiber intake to 30g per day through whole grains, fruits, and vegetables. Add fermented foods like yogurt, kefir, and sauerkraut to your diet. Reduce processed sugar consumption to less than 25g per day.",
                  },
                  {
                    date: "2023-03-15",
                    recommendation:
                      "Consider a high-quality probiotic supplement with multiple strains. Aim for 7-8 hours of quality sleep each night. Practice stress-reduction techniques like meditation or deep breathing.",
                  },
                ].map((rec, i) => (
                  <div key={i} className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between">
                      <p className="font-medium">Recommendation</p>
                      <p className="text-sm text-muted-foreground">{new Date(rec.date).toLocaleDateString()}</p>
                    </div>
                    <p className="text-sm">{rec.recommendation}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Send New Recommendation</CardTitle>
              <CardDescription>Provide personalized health recommendations to the patient</CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter your recommendation here..."
                className="min-h-[150px]"
                value={recommendation}
                onChange={(e) => setRecommendation(e.target.value)}
              />
            </CardContent>
            <CardFooter>
              <Button onClick={handleSendRecommendation} disabled={isSending}>
                {isSending ? "Sending..." : "Send Recommendation"}
              </Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
