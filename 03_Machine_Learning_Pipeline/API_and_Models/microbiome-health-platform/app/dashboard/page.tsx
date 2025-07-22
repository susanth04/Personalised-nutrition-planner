import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart3, FileUp, ClipboardList, Users } from "lucide-react"
import DigitalTwinView from "@/components/DigitalTwinView"
import TreatmentOptimizer from "@/components/TreatmentOptimizer"

export default function DashboardPage() {
  // For development, show both dashboards
  return (
    <Tabs defaultValue="patient" className="space-y-4">
      <TabsList>
        <TabsTrigger value="patient">Patient Dashboard</TabsTrigger>
        <TabsTrigger value="doctor">Doctor Dashboard</TabsTrigger>
      </TabsList>
      <TabsContent value="patient">
        <PatientDashboard />
      </TabsContent>
      <TabsContent value="doctor">
        <DoctorDashboard />
      </TabsContent>
    </Tabs>
  )
}

function PatientDashboard() {
  // Mock patient data that would normally come from a database
  const patientData = {
    id: "patient-123",
    age: 42,
    weight: 75,
    height: 175,
    symptoms: {
      bloating: 4,
      abdominal_pain: 2,
      diarrhea: 1,
      constipation: 3
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome to your health dashboard. Monitor your microbiome health and insights.
        </p>
      </div>
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="digital-twin">Digital Twin</TabsTrigger>
          <TabsTrigger value="activity">Activity</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Data Uploads</CardTitle>
                <FileUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">3</div>
                <p className="text-xs text-muted-foreground">+1 from last month</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Symptom Logs</CardTitle>
                <ClipboardList className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">12</div>
                <p className="text-xs text-muted-foreground">+4 from last month</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Health Insights</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">5</div>
                <p className="text-xs text-muted-foreground">+2 from last month</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Doctor Recommendations</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">2</div>
                <p className="text-xs text-muted-foreground">+1 from last month</p>
              </CardContent>
            </Card>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Microbiome Diversity</CardTitle>
                <CardDescription>Your microbiome diversity over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px] bg-muted rounded-md flex items-center justify-center">
                  <p className="text-muted-foreground">Diversity Chart</p>
                </div>
              </CardContent>
            </Card>
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Recent Symptoms</CardTitle>
                <CardDescription>Your recently reported symptoms</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {["Bloating", "Fatigue", "Headache"].map((symptom, i) => (
                    <div key={i} className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-primary mr-2"></div>
                      <div className="flex-1 space-y-1">
                        <p className="text-sm font-medium leading-none">{symptom}</p>
                        <p className="text-sm text-muted-foreground">
                          {new Date(Date.now() - i * 86400000).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Health Insights</CardTitle>
              <CardDescription>Personalized insights based on your microbiome data</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Gut Health Score: 78/100</h3>
                <p className="text-sm text-muted-foreground">
                  Your gut health is above average. Continue with your current diet and consider adding more fermented
                  foods.
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Microbiome Diversity</h3>
                <p className="text-sm text-muted-foreground">
                  Your microbiome diversity is in the 65th percentile compared to others your age.
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-medium">Recommendations</h3>
                <ul className="list-disc pl-5 text-sm text-muted-foreground space-y-1">
                  <li>Increase fiber intake to 30g per day</li>
                  <li>Add probiotic-rich foods to your diet</li>
                  <li>Consider reducing processed sugar consumption</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="digital-twin" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-1">
            <DigitalTwinView patientId={patientData.id} patientData={patientData} />
          </div>
        </TabsContent>
        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Your recent actions and updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { action: "Uploaded microbiome data", date: "2 days ago" },
                  { action: "Logged new symptoms", date: "3 days ago" },
                  { action: "Received doctor recommendation", date: "1 week ago" },
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
        </TabsContent>
      </Tabs>
    </div>
  )
}

function DoctorDashboard() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Doctor Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome to your dashboard. Manage patient data and provide health insights.
        </p>
      </div>
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="patients">Patients</TabsTrigger>
          <TabsTrigger value="treatment">Treatment</TabsTrigger>
          <TabsTrigger value="research">Research</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Patients</CardTitle>
                <Users className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">24</div>
                <p className="text-xs text-muted-foreground">+3 from last month</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Pending Reviews</CardTitle>
                <ClipboardList className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">7</div>
                <p className="text-xs text-muted-foreground">+2 from yesterday</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Recommendations Sent</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">18</div>
                <p className="text-xs text-muted-foreground">+5 from last month</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Data Analyses</CardTitle>
                <FileUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">32</div>
                <p className="text-xs text-muted-foreground">+8 from last month</p>
              </CardContent>
            </Card>
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
            <Card className="col-span-4">
              <CardHeader>
                <CardTitle>Patient Activity</CardTitle>
                <CardDescription>Patient data uploads and symptom logs</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-[300px] bg-muted rounded-md flex items-center justify-center">
                  <p className="text-muted-foreground">Activity Chart</p>
                </div>
              </CardContent>
            </Card>
            <Card className="col-span-3">
              <CardHeader>
                <CardTitle>Recent Patients</CardTitle>
                <CardDescription>Patients who recently uploaded data</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {["John Doe", "Jane Smith", "Robert Johnson"].map((name, i) => (
                    <div key={i} className="flex items-center">
                      <div className="w-2 h-2 rounded-full bg-primary mr-2"></div>
                      <div className="flex-1 space-y-1">
                        <p className="text-sm font-medium leading-none">{name}</p>
                        <p className="text-sm text-muted-foreground">
                          {new Date(Date.now() - i * 86400000).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="patients" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Patient List</CardTitle>
              <CardDescription>Manage your patients and their data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { name: "John Doe", email: "john@example.com", lastUpload: "2 days ago" },
                  { name: "Jane Smith", email: "jane@example.com", lastUpload: "1 week ago" },
                  { name: "Robert Johnson", email: "robert@example.com", lastUpload: "3 days ago" },
                  { name: "Emily Davis", email: "emily@example.com", lastUpload: "1 day ago" },
                ].map((patient, i) => (
                  <div key={i} className="flex items-center justify-between p-2 hover:bg-muted rounded-md">
                    <div className="space-y-1">
                      <p className="font-medium">{patient.name}</p>
                      <p className="text-sm text-muted-foreground">{patient.email}</p>
                    </div>
                    <div className="text-sm text-muted-foreground">Last upload: {patient.lastUpload}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="treatment" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-1">
            <TreatmentOptimizer 
              patientId="patient-123" 
              patientMetrics={{
                inflammation: 1.8,
                bloating: 7,
                gut_permeability: 0.6,
                butyrate_production: 0.4,
                microbiome_diversity: 0.5
              }} 
            />
          </div>
        </TabsContent>
        <TabsContent value="research" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Data Analysis</CardTitle>
              <CardDescription>Analyze patient microbiome data</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="h-[300px] bg-muted rounded-md flex items-center justify-center">
                  <p className="text-muted-foreground">Analysis Dashboard</p>
                </div>
                <p className="text-sm text-muted-foreground">
                  Use the analysis tools to identify patterns in patient microbiome data and provide personalized health
                  recommendations.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
