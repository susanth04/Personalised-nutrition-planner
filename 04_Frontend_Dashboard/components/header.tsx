"use client"

"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X, LogIn } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"
import { useAuth } from "@/lib/auth-context"
import { useLogout } from "@/lib/auth-hooks"
import { useRouter } from "next/navigation"

export function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const pathname = usePathname()
  const { user, userProfile, loading } = useAuth()
  const { logout } = useLogout()
  const router = useRouter()

  const handleLogout = async () => {
    try {
      await logout()
      router.push("/login")
    } catch (error) {
      console.error("Logout failed:", error)
    }
  }

  const navigation = [
    { name: "Home", href: "/" },
    { name: "Meal Plan", href: "/meal-plan" },
    { name: "Digital Twin", href: "/digital-twin" },
    { name: "About", href: "/about" },
  ]

  const isActive = (path: string) => pathname === path

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex items-center justify-between h-16 px-4 mx-auto max-w-7xl">
        <div className="flex items-center gap-2">
          <Link href="/" className="flex items-center gap-2">
            <span className="text-xl font-bold text-emerald-600 dark:text-emerald-500">NutriTwin</span>
          </Link>
        </div>

        {/* Desktop navigation */}
        <nav className="hidden md:flex md:items-center md:gap-6">
          {navigation.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`text-sm font-medium transition-colors ${
                isActive(item.href) ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {item.name}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <ModeToggle />

          {/* Auth buttons (Desktop) */}
          <div className="hidden md:flex md:items-center md:gap-2">
            {!loading && user ? (
              <>
                <span className="text-sm text-muted-foreground mr-2">
                  {userProfile?.firstName || user.email}
                </span>
                <Button variant="ghost" size="sm" onClick={handleLogout}>
                  Logout
                </Button>
              </>
            ) : !loading ? (
              <>
                <Button asChild variant="ghost" size="sm">
                  <Link href="/login">Login</Link>
                </Button>
                <Button asChild size="sm">
                  <Link href="/register">Sign Up</Link>
                </Button>
              </>
            ) : null}
          </div>

          {/* Mobile menu button */}
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            aria-label="Toggle menu"
          >
            
            {/* Mobile Auth buttons */}
            <div className="border-t mt-3 pt-3 space-y-2">
              {!loading && user ? (
                <>
                  <div className="px-3 py-2 text-sm font-medium text-muted-foreground">
                    {userProfile?.firstName || user.email}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full justify-start"
                    onClick={() => {
                      handleLogout()
                      setIsMenuOpen(false)
                    }}
                  >
                    Logout
                  </Button>
                </>
              ) : !loading ? (
                <>
                  <Link
                    href="/login"
                    className="block px-3 py-2 text-base font-medium rounded-md text-muted-foreground hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-foreground"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    Login
                  </Link>
                  <Link
                    href="/register"
                    className="block px-3 py-2 text-base font-medium rounded-md bg-emerald-600 text-white hover:bg-emerald-700"
                    onClick={() => setIsMenuOpen(false)}
                  >
                    Sign Up
                  </Link>
                </>
              ) : null}
            </div>
            {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </Button>
        </div>
      </div>

      {/* Mobile navigation */}
      {isMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 border-t">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={`block px-3 py-2 text-base font-medium rounded-md ${
                  isActive(item.href)
                    ? "bg-gray-100 dark:bg-gray-800 text-foreground"
                    : "text-muted-foreground hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-foreground"
                }`}
                onClick={() => setIsMenuOpen(false)}
              >
                {item.name}
              </Link>
            ))}
          </div>
        </div>
      )}
    </header>
  )
}
