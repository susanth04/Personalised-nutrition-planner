"use client"

import type React from "react"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { getSupabaseClient } from "@/lib/supabase"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function LoginForm() {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showResendConfirmation, setShowResendConfirmation] = useState(false)
  const [resendLoading, setResendLoading] = useState(false)
  const [resendSuccess, setResendSuccess] = useState(false)
  const router = useRouter()
  const supabase = getSupabaseClient()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setShowResendConfirmation(false)

    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })

      if (error) {
        // Check if the error is related to email confirmation
        if (error.message.includes("Email not confirmed") || error.message.includes("Invalid login credentials")) {
          setShowResendConfirmation(true)
          throw new Error("Email not confirmed. Please check your inbox or request a new confirmation email.")
        }
        throw error
      }

      // Redirect based on user type
      const userType = data.user?.user_metadata?.user_type || "patient"
      router.push(userType === "doctor" ? "/doctor/dashboard" : "/patient/dashboard")
    } catch (err: any) {
      setError(err.message || "An error occurred during login")
    } finally {
      setLoading(false)
    }
  }

  const handleResendConfirmation = async () => {
    setResendLoading(true)
    try {
      const { error } = await supabase.auth.resend({
        type: "signup",
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      })

      if (error) throw error
      setResendSuccess(true)
    } catch (err: any) {
      setError(err.message || "Failed to resend confirmation email")
    } finally {
      setResendLoading(false)
    }
  }

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Log in</CardTitle>
        <CardDescription>Access your microbiome analysis platform account</CardDescription>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {resendSuccess && (
          <Alert className="mb-4 bg-green-50 text-green-800 border-green-200">
            <AlertDescription>Confirmation email sent! Please check your inbox.</AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleLogin}>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Logging in..." : "Log In"}
            </Button>

            {showResendConfirmation && (
              <Button
                type="button"
                variant="outline"
                className="w-full mt-2"
                onClick={handleResendConfirmation}
                disabled={resendLoading}
              >
                {resendLoading ? "Sending..." : "Resend confirmation email"}
              </Button>
            )}
          </div>
        </form>
      </CardContent>
      <CardFooter className="flex justify-between">
        <a href="/forgot-password" className="text-sm text-primary hover:underline">
          Forgot password?
        </a>
        <a href="/signup" className="text-sm text-primary hover:underline">
          Create an account
        </a>
      </CardFooter>
    </Card>
  )
}
