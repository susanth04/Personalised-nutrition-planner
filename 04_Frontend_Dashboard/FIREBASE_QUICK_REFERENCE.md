# Firebase Authentication - Quick Reference

## Installation

```bash
npm install firebase
```

## Core Files

| File | Purpose |
|------|---------|
| `lib/firebase.ts` | Firebase initialization with config |
| `lib/auth-service.ts` | Authentication business logic |
| `lib/firestore-service.ts` | User data operations |
| `lib/auth-context.tsx` | React Context provider |
| `lib/auth-hooks.ts` | Custom React hooks |
| `lib/protected-route.tsx` | Route protection utility |

## Key Functions

### Authentication

```typescript
import { 
  registerUser, 
  loginUser, 
  logoutUser, 
  resetPassword,
  getCurrentUser 
} from '@/lib/auth-service';

// Register new user
await registerUser(email, password, firstName, lastName, role)

// Login user
await loginUser(email, password)

// Logout user
await logoutUser()

// Send password reset email
await resetPassword(email)

// Get current authenticated user
const user = getCurrentUser()
```

### Firestore User Data

```typescript
import {
  setUserData,
  getUserData,
  updateUserData,
  deleteUserData,
  getUserByEmail,
  getUsersByRole
} from '@/lib/firestore-service';

// Store user data
await setUserData(uid, userData)

// Retrieve user data
const user = await getUserData(uid)

// Update user data
await updateUserData(uid, userData)

// Delete user data
await deleteUserData(uid)

// Query by email
const user = await getUserByEmail(email)

// Query by role
const users = await getUsersByRole('nutritionist')
```

## React Hooks

```typescript
import { useAuth } from '@/lib/auth-context';
import { useLogin, useRegister, useLogout, usePasswordReset } from '@/lib/auth-hooks';

// Get auth state in any component
const { user, userProfile, loading, isAuthenticated } = useAuth();

// Registration
const { register, loading, error } = useRegister({
  onSuccess: (user) => { /* ... */ },
  onError: (error) => { /* ... */ }
});
await register(email, password, firstName, lastName, role);

// Login
const { login, loading, error } = useLogin({
  onSuccess: (user) => { /* ... */ }
});
await login(email, password);

// Logout
const { logout, loading, error } = useLogout();
await logout();

// Password reset
const { reset, loading, error, success } = usePasswordReset({
  onSuccess: () => { /* ... */ }
});
await reset(email);
```

## Protected Pages

```typescript
'use client';

import { useAuth } from '@/lib/auth-context';
import { useRouter } from 'next/navigation';

export default function ProtectedPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  if (!loading && !user) {
    router.push('/login');
    return null;
  }

  return <div>Protected content</div>;
}
```

## URLs

- Login: `/login`
- Register: `/register`
- Forgot Password: `/forgot-password`
- Dashboard: `/dashboard`

## Firestore Database Schema

```javascript
users/
  {uid}/
    - uid: string
    - email: string
    - firstName: string
    - lastName: string
    - role: string
    - createdAt: Timestamp
    - updatedAt: Timestamp
```

## Common Patterns

### Form with Registration Hook

```typescript
const { register, loading, error } = useRegister({
  onSuccess: () => router.push('/dashboard')
});

const handleSubmit = async (e) => {
  e.preventDefault();
  try {
    await register(email, password, firstName, lastName, role);
  } catch (err) {
    console.error(err);
  }
};
```

### Protected Route Component

```typescript
export default function Dashboard() {
  const { user, loading } = useAuth();

  if (loading) return <Spinner />;
  if (!user) return <Redirect to="/login" />;

  return <div>Welcome {user.email}</div>;
}
```

### User Profile Display

```typescript
const { userProfile } = useAuth();

return (
  <div>
    <h1>{userProfile?.firstName} {userProfile?.lastName}</h1>
    <p>{userProfile?.email}</p>
    <p>Role: {userProfile?.role}</p>
  </div>
);
```

## Error Handling

```typescript
try {
  await loginUser(email, password);
} catch (error) {
  // Error messages are user-friendly
  console.error(error.message);
  // Examples:
  // "Invalid email or password."
  // "Email already registered."
  // "Password must be at least 6 characters"
}
```

## Firebase Console Links

- [Firebase Console](https://console.firebase.google.com/)
- Project: `nutritwin-cd71b`
- Authentication: https://console.firebase.google.com/u/0/project/nutritwin-cd71b/authentication/users
- Firestore: https://console.firebase.google.com/u/0/project/nutritwin-cd71b/firestore/data

## Security Rules Template

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{uid} {
      allow read, write: if request.auth.uid == uid;
    }
  }
}
```

## Environment Setup

No environment variables needed! Firebase credentials are public and configured in `lib/firebase.ts`.

---

See `FIREBASE_INTEGRATION.md` for complete API documentation.
