"use client"

import { useState, useEffect } from "react"
import { CheckCircle, XCircle, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { apiClient } from "@/lib/api-client"

export function ApiStatus() {
  const [status, setStatus] = useState<"loading" | "online" | "offline">("loading")

  useEffect(() => {
    const checkApiStatus = async () => {
      try {
        await apiClient.get("/health")
        setStatus("online")
      } catch (error) {
        setStatus("offline")
      }
    }

    checkApiStatus()
  }, [])

  if (status === "loading") {
    return (
      <Alert className="max-w-md">
        <AlertCircle className="w-4 h-4 text-gray-500" />
        <AlertTitle>Checking API Status</AlertTitle>
        <AlertDescription>Connecting to the Personalized Nutrition API...</AlertDescription>
      </Alert>
    )
  }

  if (status === "offline") {
    return (
      <Alert variant="destructive" className="max-w-md">
        <XCircle className="w-4 h-4" />
        <AlertTitle>API Offline</AlertTitle>
        <AlertDescription>
          The Personalized Nutrition API is currently unavailable. Some features may not work.
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Alert className="max-w-md bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900">
      <CheckCircle className="w-4 h-4 text-emerald-600 dark:text-emerald-500" />
      <AlertTitle className="text-emerald-800 dark:text-emerald-400">API Online</AlertTitle>
      <AlertDescription className="text-emerald-700 dark:text-emerald-300">
        Connected to the Personalized Nutrition API. All features are available.
      </AlertDescription>
    </Alert>
  )
}
