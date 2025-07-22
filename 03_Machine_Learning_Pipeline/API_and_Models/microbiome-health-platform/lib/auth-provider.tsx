"use client"

import type React from "react"
import { createContext, useContext, useState } from "react"
import { useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"

type User = {
  id: string
  email: string
  role: "patient" | "doctor"
  name?: string
}

type AuthContextType = {
  user: User | null
  loading: boolean
  signIn: (email: string, password: string) => Promise<{ error: any }>
  signUp: (email: string, password: string, role: "patient" | "doctor", name: string) => Promise<{ error: any }>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Mock user for development
const mockUser: User = {
  id: 'mock-user-id',
  email: 'user@example.com',
  role: 'doctor',
  name: 'Mock User'
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(mockUser)
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const signUp = async (email: string, password: string, role: "patient" | "doctor", name: string) => {
    // Mock signup functionality
    setUser({
      id: 'new-user-id',
      email,
      role,
      name
    })
    return { error: null }
  }

  const signIn = async (email: string, password: string) => {
    // Mock signin functionality
    setUser(mockUser)
    return { error: null }
  }

  const signOut = async () => {
    // Mock signout functionality
    setUser(null)
    router.push("/")
  }

  const value = {
    user,
    loading,
    signIn,
    signUp,
    signOut,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
