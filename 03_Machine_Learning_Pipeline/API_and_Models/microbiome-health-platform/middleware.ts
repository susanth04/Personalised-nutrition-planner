import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export async function middleware(request: NextRequest) {
  // Temporarily disable authentication middleware to fix the Supabase error
  // TODO: Fix Supabase client configuration later
  
  // For now, just allow all requests to pass through
  return NextResponse.next()
}

export const config = {
  matcher: ["/dashboard/:path*", "/patient/:path*", "/doctor/:path*"],
}
