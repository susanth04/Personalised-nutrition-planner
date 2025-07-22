"use client"

import { useState } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-provider"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Search, Filter, ChevronRight } from "lucide-react"

export default function PatientsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const { user } = useAuth()
  const router = useRouter()

  // Sample patient data
  const patients = [
    {
      id: "1",
      name: "John Doe",
      email: "john@example.com",
      age: 35,
      lastUpload: "2023-04-08",
      healthScore: 78,
      status: "Active",
    },
    {
      id: "2",
      name: "Jane Smith",
      email: "jane@example.com",
      age: 42,
      lastUpload: "2023-04-05",
      healthScore: 65,
      status: "Needs Review",
    },
    {
      id: "3",
      name: "Robert Johnson",
      email: "robert@example.com",
      age: 28,
      lastUpload: "2023-04-01",
      healthScore: 82,
      status: "Active",
    },
    {
      id: "4",
      name: "Emily Davis",
      email: "emily@example.com",
      age: 51,
      lastUpload: "2023-03-28",
      healthScore: 70,
      status: "Needs Review",
    },
    {
      id: "5",
      name: "Michael Wilson",
      email: "michael@example.com",
      age: 45,
      lastUpload: "2023-03-25",
      healthScore: 75,
      status: "Active",
    },
  ]

  // Filter patients based on search query
  const filteredPatients = patients.filter(
    (patient) =>
      patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      patient.email.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Patients</h1>
        <p className="text-muted-foreground">Manage your patients and their microbiome data</p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Patient List</CardTitle>
          <CardDescription>View and manage all your patients</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  type="search"
                  placeholder="Search patients..."
                  className="pl-8"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <Button variant="outline" size="icon">
                <Filter className="h-4 w-4" />
                <span className="sr-only">Filter</span>
              </Button>
              <Button>Add Patient</Button>
            </div>
            <div className="rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Age</TableHead>
                    <TableHead>Last Upload</TableHead>
                    <TableHead>Health Score</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredPatients.map((patient) => (
                    <TableRow key={patient.id}>
                      <TableCell className="font-medium">
                        <div>
                          {patient.name}
                          <div className="text-sm text-muted-foreground">{patient.email}</div>
                        </div>
                      </TableCell>
                      <TableCell>{patient.age}</TableCell>
                      <TableCell>{new Date(patient.lastUpload).toLocaleDateString()}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <div
                            className="h-2 w-16 rounded-full bg-muted overflow-hidden"
                            role="progressbar"
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={patient.healthScore}
                          >
                            <div
                              className={`h-full ${
                                patient.healthScore >= 80
                                  ? "bg-green-500"
                                  : patient.healthScore >= 60
                                    ? "bg-yellow-500"
                                    : "bg-red-500"
                              }`}
                              style={{ width: `${patient.healthScore}%` }}
                            />
                          </div>
                          <span className="text-sm">{patient.healthScore}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant={patient.status === "Active" ? "default" : "secondary"}>{patient.status}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Link href={`/dashboard/patients/${patient.id}`} passHref>
                          <Button variant="ghost" size="icon">
                            <ChevronRight className="h-4 w-4" />
                            <span className="sr-only">View patient</span>
                          </Button>
                        </Link>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
