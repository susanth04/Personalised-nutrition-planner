// Mock server-side client that doesn't require Supabase
export function createClient() {
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
