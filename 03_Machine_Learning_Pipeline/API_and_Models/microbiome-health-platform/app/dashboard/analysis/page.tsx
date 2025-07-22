"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts"

export default function AnalysisPage() {
  const [selectedPatient, setSelectedPatient] = useState("all")
  const [selectedAnalysis, setSelectedAnalysis] = useState("lda")

  // Sample data for charts
  const ldaTopics = [
    { name: "Topic 1", bacteria1: 30, bacteria2: 45, bacteria3: 25 },
    { name: "Topic 2", bacteria1: 20, bacteria2: 35, bacteria3: 45 },
    { name: "Topic 3", bacteria1: 40, bacteria2: 25, bacteria3: 35 },
    { name: "Topic 4", bacteria1: 35, bacteria2: 30, bacteria3: 35 },
    { name: "Topic 5", bacteria1: 25, bacteria2: 40, bacteria3: 35 },
  ]

  const clusterData = [
    { x: 65, y: 78, z: 10, name: "Patient 1" },
    { x: 72, y: 65, z: 8, name: "Patient 2" },
    { x: 83, y: 70, z: 12, name: "Patient 3" },
    { x: 58, y: 85, z: 15, name: "Patient 4" },
    { x: 90, y: 92, z: 20, name: "Patient 5" },
    { x: 75, y: 68, z: 9, name: "Patient 6" },
    { x: 62, y: 73, z: 11, name: "Patient 7" },
  ]

  const correlationData = [
    { name: "Bloating", bacteria1: 65, bacteria2: 35, bacteria3: 45 },
    { name: "Fatigue", bacteria1: 45, bacteria2: 55, bacteria3: 30 },
    { name: "Headache", bacteria1: 30, bacteria2: 40, bacteria3: 60 },
    { name: "Joint Pain", bacteria1: 25, bacteria2: 65, bacteria3: 40 },
    { name: "Skin Issues", bacteria1: 55, bacteria2: 30, bacteria3: 50 },
  ]

  const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Data Analysis</h1>
        <p className="text-muted-foreground">Analyze microbiome data using advanced techniques</p>
      </div>
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <Select value={selectedPatient} onValueChange={setSelectedPatient}>
            <SelectTrigger>
              <SelectValue placeholder="Select Patient" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Patients</SelectItem>
              <SelectItem value="john">John Doe</SelectItem>
              <SelectItem value="jane">Jane Smith</SelectItem>
              <SelectItem value="robert">Robert Johnson</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Button>Run Analysis</Button>
      </div>
      <Tabs defaultValue="lda" className="space-y-4" onValueChange={setSelectedAnalysis}>
        <TabsList>
          <TabsTrigger value="lda">LDA Analysis</TabsTrigger>
          <TabsTrigger value="clustering">Clustering</TabsTrigger>
          <TabsTrigger value="correlation">Correlation</TabsTrigger>
        </TabsList>
        <TabsContent value="lda" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Latent Dirichlet Allocation (LDA) Analysis</CardTitle>
              <CardDescription>Topic modeling to identify patterns in microbiome data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-[400px]">
                <ChartContainer
                  config={{
                    bacteria1: {
                      label: "Bacteroidetes",
                      color: "hsl(var(--chart-1))",
                    },
                    bacteria2: {
                      label: "Firmicutes",
                      color: "hsl(var(--chart-2))",
                    },
                    bacteria3: {
                      label: "Proteobacteria",
                      color: "hsl(var(--chart-3))",
                    },
                  }}
                  className="h-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={ldaTopics}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend />
                      <Bar dataKey="bacteria1" fill="var(--color-bacteria1)" />
                      <Bar dataKey="bacteria2" fill="var(--color-bacteria2)" />
                      <Bar dataKey="bacteria3" fill="var(--color-bacteria3)" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">LDA Topic Interpretation</h3>
                <div className="space-y-2">
                  <div className="border rounded-md p-3">
                    <p className="font-medium">Topic 1: Gut Health</p>
                    <p className="text-sm text-muted-foreground">
                      Dominated by Bacteroidetes and Firmicutes, associated with good gut health and digestion.
                    </p>
                  </div>
                  <div className="border rounded-md p-3">
                    <p className="font-medium">Topic 2: Inflammation</p>
                    <p className="text-sm text-muted-foreground">
                      Higher levels of Proteobacteria, often associated with inflammation and digestive issues.
                    </p>
                  </div>
                  <div className="border rounded-md p-3">
                    <p className="font-medium">Topic 3: Immune Function</p>
                    <p className="text-sm text-muted-foreground">
                      Balanced distribution of bacteria types, associated with healthy immune function.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="clustering" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Patient Clustering</CardTitle>
              <CardDescription>Clustering patients based on microbiome profiles</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" dataKey="x" name="Diversity" unit="%" />
                    <YAxis type="number" dataKey="y" name="Health Score" unit="" />
                    <ZAxis type="number" dataKey="z" range={[60, 400]} name="Age" />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Legend />
                    <Scatter name="Patients" data={clusterData} fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Cluster Analysis</h3>
                <p className="text-sm">
                  The scatter plot shows patient clustering based on microbiome diversity (x-axis) and overall health
                  score (y-axis). The size of each point represents the patient's age. We can identify three main
                  clusters:
                </p>
                <ul className="list-disc pl-5 text-sm space-y-1">
                  <li>Cluster 1: High diversity, high health score (upper right)</li>
                  <li>Cluster 2: Medium diversity, medium health score (center)</li>
                  <li>Cluster 3: Low diversity, variable health score (left)</li>
                </ul>
                <p className="text-sm">
                  Patients in Cluster 1 generally show better health outcomes and fewer reported symptoms.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="correlation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Symptom-Bacteria Correlation</CardTitle>
              <CardDescription>Correlation between reported symptoms and bacterial composition</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="h-[400px]">
                <ChartContainer
                  config={{
                    bacteria1: {
                      label: "Bacteroidetes",
                      color: "hsl(var(--chart-1))",
                    },
                    bacteria2: {
                      label: "Firmicutes",
                      color: "hsl(var(--chart-2))",
                    },
                    bacteria3: {
                      label: "Proteobacteria",
                      color: "hsl(var(--chart-3))",
                    },
                  }}
                  className="h-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={correlationData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Legend />
                      <Bar dataKey="bacteria1" fill="var(--color-bacteria1)" />
                      <Bar dataKey="bacteria2" fill="var(--color-bacteria2)" />
                      <Bar dataKey="bacteria3" fill="var(--color-bacteria3)" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartContainer>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Key Correlations</h3>
                <ul className="list-disc pl-5 text-sm space-y-1">
                  <li>Bloating shows strong correlation with elevated Bacteroidetes</li>
                  <li>Fatigue correlates with higher Firmicutes levels</li>
                  <li>Headache and joint pain show correlation with Proteobacteria</li>
                  <li>Skin issues correlate with imbalance between Bacteroidetes and Proteobacteria</li>
                </ul>
                <p className="text-sm text-muted-foreground mt-2">
                  These correlations can help guide personalized dietary and lifestyle recommendations for patients.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
