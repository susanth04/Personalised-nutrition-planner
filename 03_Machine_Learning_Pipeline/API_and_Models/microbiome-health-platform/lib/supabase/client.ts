"use client"

// Mock client that doesn't require Supabase
export const createClient = () => {
  return {
    auth: {
      getSession: async () => ({
        data: { 
          session: {
            user: {
              id: 'mock-user-id',
              email: 'user@example.com'
            }
          }
        }
      }),
      signInWithPassword: async () => ({
        data: {
          user: {
            id: 'mock-user-id',
            email: 'user@example.com'
          }
        },
        error: null
      }),
      signUp: async () => ({
        data: {
          user: {
            id: 'mock-user-id',
            email: 'user@example.com'
          }
        },
        error: null
      }),
      signOut: async () => ({}),
      onAuthStateChange: () => ({
        data: { 
          subscription: {
            unsubscribe: () => {}
          }
        }
      })
    },
    from: (table) => ({
      select: () => ({
        eq: () => ({
          single: async () => ({
            data: {
              id: 'mock-user-id',
              email: 'user@example.com',
              role: 'doctor',
              name: 'Mock User'
            },
            error: null
          })
        })
      }),
      insert: async () => ({ error: null })
    })
  }
}
