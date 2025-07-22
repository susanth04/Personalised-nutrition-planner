"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useAuth } from "@/lib/auth-provider"
import { cn } from "@/lib/utils"
import { BarChart3, FileUp, Home, ClipboardList, Users, User, Settings } from "lucide-react"

export function DashboardSidebar() {
  const pathname = usePathname()
  const { user } = useAuth()

  const patientLinks = [
    {
      title: "Dashboard",
      href: "/dashboard",
      icon: Home,
    },
    {
      title: "Upload Data",
      href: "/dashboard/upload",
      icon: FileUp,
    },
    {
      title: "Symptoms",
      href: "/dashboard/symptoms",
      icon: ClipboardList,
    },
    {
      title: "Health Insights",
      href: "/dashboard/insights",
      icon: BarChart3,
    },
    {
      title: "Profile",
      href: "/dashboard/profile",
      icon: User,
    },
    {
      title: "Settings",
      href: "/dashboard/settings",
      icon: Settings,
    },
  ]

  const doctorLinks = [
    {
      title: "Dashboard",
      href: "/dashboard",
      icon: Home,
    },
    {
      title: "Patients",
      href: "/dashboard/patients",
      icon: Users,
    },
    {
      title: "Data Analysis",
      href: "/dashboard/analysis",
      icon: BarChart3,
    },
    {
      title: "Profile",
      href: "/dashboard/profile",
      icon: User,
    },
    {
      title: "Settings",
      href: "/dashboard/settings",
      icon: Settings,
    },
  ]

  const links = user?.role === "doctor" ? doctorLinks : patientLinks

  return (
    <div className="hidden border-r bg-muted/40 md:block md:w-64">
      <div className="flex h-full max-h-screen flex-col gap-2">
        <div className="flex-1 overflow-auto py-2">
          <nav className="grid items-start px-2 text-sm font-medium">
            {links.map((link, index) => (
              <Link
                key={index}
                href={link.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground hover:text-foreground",
                  pathname === link.href && "bg-muted text-foreground",
                )}
              >
                <link.icon className="h-4 w-4" />
                {link.title}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </div>
  )
}
