import axios from "axios"

// Create an axios instance with default config
export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
})

// Add a response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle specific error codes
    if (error.response) {
      switch (error.response.status) {
        case 500:
          console.error("Server error:", error.response.data.detail || "Unknown server error")
          break
        default:
          console.error("API error:", error.response.data)
      }
    } else if (error.request) {
      console.error("No response received from API")
    } else {
      console.error("Error setting up request:", error.message)
    }

    return Promise.reject(error)
  },
)
