"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts"

export default function InsightsPage() {
  const [selectedUpload, setSelectedUpload] = useState("latest")

  // Sample data for charts
  const diversityData = [
    { name: "Bacteroidetes", value: 35 },
    { name: "Firmicutes", value: 30 },
    { name: "Proteobacteria", value: 15 },
    { name: "Actinobacteria", value: 10 },
    { name: "Other", value: 10 },
  ]

  const timeSeriesData = [
    { date: "Jan", diversity: 65, health: 70 },
    { date: "Feb", diversity: 68, health: 72 },
    { date: "Mar", diversity: 75, health: 78 },
    { date: "Apr", diversity: 72, health: 75 },
    { date: "May", diversity: 78, health: 80 },
    { date: "Jun", diversity: 82, health: 85 },
  ]

  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Health Insights</h1>
        <p className="text-muted-foreground">Personalized insights based on your microbiome data</p>
      </div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="col-span-full">
          <CardHeader>
            <CardTitle>Summary</CardTitle>
            <CardDescription>Overview of your microbiome health</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2 border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Gut Health Score</p>
                  <p className="text-3xl font-bold">78/100</p>
                  <p className="text-xs text-muted-foreground">↑ 5 points from last analysis</p>
                </div>
                <div className="space-y-2 border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Diversity Index</p>
                  <p className="text-3xl font-bold">3.8/5</p>
                  <p className="text-xs text-muted-foreground">↑ 0.3 from last analysis</p>
                </div>
                <div className="space-y-2 border rounded-lg p-4">
                  <p className="text-sm text-muted-foreground">Inflammation Markers</p>
                  <p className="text-3xl font-bold">Low</p>
                  <p className="text-xs text-muted-foreground">Improved from moderate</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Microbiome Diversity Over Time</CardTitle>
            <CardDescription>Tracking changes in your microbiome diversity</CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                diversity: {
                  label: "Diversity Index",
                  color: "hsl(var(--chart-1))",
                },
                health: {
                  label: "Health Score",
                  color: "hsl(var(--chart-2))",
                },
              }}
              className="h-[300px]"
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Legend />
                  <Line type="monotone" dataKey="diversity" stroke="var(--color-diversity)" strokeWidth={2} />
                  <Line type="monotone" dataKey="health" stroke="var(--color-health)" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Microbiome Composition</CardTitle>
            <CardDescription>Breakdown of your gut bacteria</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={diversityData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {diversityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
      <Tabs defaultValue="recommendations" className="space-y-4">
        <TabsList>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="analysis">Detailed Analysis</TabsTrigger>
          <TabsTrigger value="history">Historical Data</TabsTrigger>
        </TabsList>
        <TabsContent value="recommendations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Personalized Recommendations</CardTitle>
              <CardDescription>Based on your microbiome data and symptoms</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Dietary Recommendations</h3>
                <ul className="list-disc pl-5 text-sm space-y-1">
                  <li>Increase fiber intake to 30g per day through whole grains, fruits, and vegetables</li>
                  <li>Add fermented foods like yogurt, kefir, and sauerkraut to your diet</li>
                  <li>Reduce processed sugar consumption to less than 25g per day</li>
                  <li>Include prebiotic foods such as garlic, onions, and leeks</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Lifestyle Recommendations</h3>
                <ul className="list-disc pl-5 text-sm space-y-1">
                  <li>Aim for 7-8 hours of quality sleep each night</li>
                  <li>Engage in moderate exercise for at least 30 minutes, 5 days a week</li>
                  <li>Practice stress-reduction techniques like meditation or deep breathing</li>
                  <li>Stay hydrated with at least 2 liters of water daily</li>
                </ul>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Supplement Recommendations</h3>
                <ul className="list-disc pl-5 text-sm space-y-1">
                  <li>Consider a high-quality probiotic supplement with multiple strains</li>
                  <li>Omega-3 fatty acids may help reduce inflammation</li>
                  <li>Vitamin D supplementation if levels are low</li>
                </ul>
                <p className="text-xs text-muted-foreground mt-2">
                  Note: Always consult with your healthcare provider before starting any new supplement regimen.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detailed Microbiome Analysis</CardTitle>
              <CardDescription>In-depth analysis of your microbiome data</CardDescription>
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
                <h3 className="text-lg font-medium">Correlation with Symptoms</h3>
                <p className="text-sm">
                  Your reported symptoms of occasional bloating and fatigue may be related to the slightly elevated
                  levels of Proteobacteria. The recommendations provided should help address these issues.
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Comparison to Population</h3>
                <p className="text-sm">
                  Your microbiome diversity is in the 65th percentile compared to others in your age group, which is
                  good. Your Firmicutes to Bacteroidetes ratio is within the healthy range.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Historical Data</CardTitle>
              <CardDescription>View your previous microbiome analyses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { date: "June 1, 2023", score: 78, diversity: 3.8 },
                  { date: "March 15, 2023", score: 73, diversity: 3.5 },
                  { date: "December 10, 2022", score: 70, diversity: 3.3 },
                ].map((entry, i) => (
                  <div key={i} className="border rounded-md p-3 space-y-2">
                    <div className="flex justify-between">
                      <p className="font-medium">{entry.date}</p>
                      <button className="text-sm text-primary">View Details</button>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <p className="text-muted-foreground">Health Score</p>
                        <p>{entry.score}/100</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Diversity</p>
                        <p>{entry.diversity}/5</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
