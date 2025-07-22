import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 bg-gradient-to-b from-background to-muted">
      <div className="w-full max-w-3xl space-y-8 text-center">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
          Microbiome Health Platform
        </h1>
        <p className="text-xl text-muted-foreground">
          Advanced digital twin technology for personalized gut microbiome health management
        </p>
        <div className="flex justify-center gap-4">
          <Link href="/dashboard">
            <Button size="lg" className="px-8">
              Enter Dashboard
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}
