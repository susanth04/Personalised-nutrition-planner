# Firebase Integration Documentation

This document explains the Firebase authentication and Firestore integration for the NutritionWin application.

## Overview

The application now uses Firebase for:
- **Authentication**: User login, registration, and password reset
- **Database**: Firestore for storing user profile information

## Firebase Configuration

The Firebase credentials are configured in [lib/firebase.ts](lib/firebase.ts):

```typescript
const firebaseConfig = {
  apiKey: "AIzaSyBRqumqQoWU5FdnuN_WTn4O0tjiCy1YZSc",
  authDomain: "nutritwin-cd71b.firebaseapp.com",
  projectId: "nutritwin-cd71b",
  storageBucket: "nutritwin-cd71b.firebasestorage.app",
  messagingSenderId: "1040958981519",
  appId: "1:1040958981519:web:e2df547f97ab68ab64b7fa",
  measurementId: "G-F5SKB6R34D"
};
```

## Architecture

### Files Structure

```
lib/
├── firebase.ts              # Firebase initialization
├── auth-service.ts          # Authentication functions
├── firestore-service.ts     # Firestore user data management
├── auth-context.tsx         # React Context for auth state
├── auth-hooks.ts            # Custom hooks for auth operations
└── protected-route.tsx      # Route protection utility

app/
├── layout.tsx               # Root layout with AuthProvider
├── login/
│   └── page.tsx            # Login page
├── register/
│   └── page.tsx            # Registration page
├── forgot-password/
│   └── page.tsx            # Password reset page
├── dashboard/
│   └── page.tsx            # Protected dashboard page
└── components/
    ├── LoginComponent.tsx
    ├── RegisterComponent.tsx
    └── ForgotPasswordComponent.tsx
```

## Authentication Services

### [auth-service.ts](lib/auth-service.ts)

Core authentication functions:

#### `registerUser(email, password, firstName, lastName, role)`
Registers a new user with email and password.

```typescript
import { registerUser } from '@/lib/auth-service';

const user = await registerUser(
  'user@example.com',
  'password123',
  'John',
  'Doe',
  'user'
);
```

#### `loginUser(email, password)`
Authenticates a user with email and password.

```typescript
import { loginUser } from '@/lib/auth-service';

const user = await loginUser('user@example.com', 'password123');
```

#### `resetPassword(email)`
Sends a password reset email to the user.

```typescript
import { resetPassword } from '@/lib/auth-service';

await resetPassword('user@example.com');
```

#### `logoutUser()`
Signs out the current user.

```typescript
import { logoutUser } from '@/lib/auth-service';

await logoutUser();
```

#### `getCurrentUser()`
Returns the current authenticated user.

```typescript
import { getCurrentUser } from '@/lib/auth-service';

const user = getCurrentUser();
if (user) {
  console.log(user.email);
}
```

#### `setupAuthStateListener(callback)`
Listens to auth state changes.

```typescript
import { setupAuthStateListener } from '@/lib/auth-service';

const unsubscribe = setupAuthStateListener((user) => {
  if (user) {
    console.log('User logged in:', user.email);
  } else {
    console.log('User logged out');
  }
});

// Clean up listener when done
unsubscribe();
```

## Firestore Services

### [firestore-service.ts](lib/firestore-service.ts)

User data management functions:

#### `setUserData(uid, userData)`
Stores user data in Firestore.

```typescript
import { setUserData } from '@/lib/firestore-service';

await setUserData(user.uid, {
  email: user.email,
  firstName: 'John',
  lastName: 'Doe',
  role: 'user'
});
```

#### `getUserData(uid)`
Retrieves user data from Firestore.

```typescript
import { getUserData } from '@/lib/firestore-service';

const userProfile = await getUserData('user-uid');
console.log(userProfile.firstName);
```

#### `updateUserData(uid, userData)`
Updates user data in Firestore.

```typescript
import { updateUserData } from '@/lib/firestore-service';

await updateUserData(user.uid, {
  firstName: 'Jane'
});
```

#### `getUserByEmail(email)`
Finds user by email address.

```typescript
import { getUserByEmail } from '@/lib/firestore-service';

const user = await getUserByEmail('user@example.com');
```

#### `getUsersByRole(role)`
Gets all users with a specific role.

```typescript
import { getUsersByRole } from '@/lib/firestore-service';

const nutritionists = await getUsersByRole('nutritionist');
```

#### `deleteUserData(uid)`
Deletes user data from Firestore.

```typescript
import { deleteUserData } from '@/lib/firestore-service';

await deleteUserData(user.uid);
```

## Custom Hooks

### [auth-hooks.ts](lib/auth-hooks.ts)

React hooks for authentication operations:

#### `useRegister(options)`
Hook for user registration.

```typescript
import { useRegister } from '@/lib/auth-hooks';

export default function RegisterPage() {
  const { register, loading, error } = useRegister({
    onSuccess: (user) => {
      console.log('Registration successful!', user.email);
    },
    onError: (error) => {
      console.error('Registration failed:', error.message);
    }
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    await register(email, password, firstName, lastName, role);
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <p>{error}</p>}
      <button disabled={loading}>
        {loading ? 'Registering...' : 'Register'}
      </button>
    </form>
  );
}
```

#### `useLogin(options)`
Hook for user login.

```typescript
import { useLogin } from '@/lib/auth-hooks';

export default function LoginPage() {
  const { login, loading, error } = useLogin({
    onSuccess: (user) => {
      router.push('/dashboard');
    }
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    await login(email, password);
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <p>{error}</p>}
      <button disabled={loading}>
        {loading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
}
```

#### `useLogout()`
Hook for user logout.

```typescript
import { useLogout } from '@/lib/auth-hooks';

export default function Dashboard() {
  const { logout, loading } = useLogout();

  const handleLogout = async () => {
    await logout();
    router.push('/login');
  };

  return <button onClick={handleLogout}>{loading ? 'Logging out...' : 'Logout'}</button>;
}
```

#### `usePasswordReset(options)`
Hook for password reset.

```typescript
import { usePasswordReset } from '@/lib/auth-hooks';

export default function ForgotPasswordPage() {
  const { reset, loading, error, success } = usePasswordReset({
    onSuccess: () => {
      console.log('Reset email sent!');
    }
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    await reset(email);
  };

  return (
    <>
      {success ? (
        <p>Check your email for reset instructions</p>
      ) : (
        <form onSubmit={handleSubmit}>
          {error && <p>{error}</p>}
          <button disabled={loading}>
            {loading ? 'Sending...' : 'Send Reset Link'}
          </button>
        </form>
      )}
    </>
  );
}
```

## React Context

### [auth-context.tsx](lib/auth-context.tsx)

Provides authentication state to the entire app.

```typescript
import { AuthProvider, useAuth } from '@/lib/auth-context';

// In your root layout:
export default function RootLayout({ children }) {
  return (
    <AuthProvider>
      {children}
    </AuthProvider>
  );
}

// In any component:
export default function MyComponent() {
  const { user, userProfile, loading, isAuthenticated } = useAuth();

  if (loading) return <p>Loading...</p>;

  if (!isAuthenticated) {
    return <p>Please log in</p>;
  }

  return (
    <div>
      <p>Welcome, {userProfile?.firstName}!</p>
      <p>Email: {user?.email}</p>
    </div>
  );
}
```

## Firestore Database Structure

### Users Collection

```
users/
├── {uid}
│   ├── uid: string
│   ├── email: string
│   ├── firstName: string
│   ├── lastName: string
│   ├── role: string (user, nutritionist, admin)
│   ├── createdAt: Timestamp
│   └── updatedAt: Timestamp
```

## User Roles

The application supports three user roles:

1. **user** - Regular user with access to their nutrition plans
2. **nutritionist** - Can create and manage nutrition plans
3. **admin** - Full access to system administration

## Protected Routes

Use the `ProtectedRoute` component to protect pages:

```typescript
import { ProtectedRoute } from '@/lib/protected-route';

export default function AdminPage() {
  return (
    <ProtectedRoute>
      <h1>Admin Dashboard</h1>
      {/* Content here is only visible to authenticated users */}
    </ProtectedRoute>
  );
}
```

Or use the `useAuth` hook for more control:

```typescript
import { useAuth } from '@/lib/auth-context';
import { useRouter } from 'next/navigation';

export default function Dashboard() {
  const { user, loading } = useAuth();
  const router = useRouter();

  if (!loading && !user) {
    router.push('/login');
    return null;
  }

  return <div>Dashboard content</div>;
}
```

## Error Handling

All authentication functions throw errors with user-friendly messages. The `getAuthErrorMessage()` function converts Firebase error codes to readable messages:

```typescript
import { loginUser, getAuthErrorMessage } from '@/lib/auth-service';

try {
  await loginUser(email, password);
} catch (error) {
  console.error(getAuthErrorMessage(error.message));
}
```

## Security Features

1. **Persistent Sessions** - Users stay logged in across page refreshes
2. **Email Verification** - Optional email verification for new accounts
3. **Password Reset** - Firebase-handled password reset with email verification
4. **Auth State Persistence** - Local storage persistence for auth tokens
5. **Secure Password Hashing** - Passwords are hashed by Firebase

## Setting Up Firestore Security Rules

In the Firebase Console, set these Firestore security rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can read and write their own data
    match /users/{uid} {
      allow read, write: if request.auth.uid == uid;
    }
  }
}
```

## Environment Variables

No environment variables are needed for the client-side Firebase initialization as the keys are public. For backend services, ensure proper API key restrictions are set in Firebase Console.

## Deployment Checklist

- [ ] Verify Firebase project is created at https://console.firebase.google.com/
- [ ] Enable Email/Password authentication in Firebase Console
- [ ] Create Firestore database
- [ ] Set up security rules for Firestore
- [ ] Install Firebase SDK: `npm install firebase`
- [ ] Test login/register/password reset flows
- [ ] Verify user data is stored in Firestore
- [ ] Test email verification emails are sent
- [ ] Set up custom domain if needed

## Troubleshooting

### Users can't sign up
- Check Firebase Email/Password auth is enabled
- Verify Firestore database exists and is accessible

### Password reset emails not sent
- Verify email is sent from Firebase Console
- Check email verification settings
- Ensure sender email is configured in Firebase

### User data not syncing
- Check Firestore security rules
- Verify user uid matches in auth and Firestore
- Check browser console for errors

## Additional Resources

- [Firebase Documentation](https://firebase.google.com/docs)
- [Firebase Auth](https://firebase.google.com/docs/auth)
- [Cloud Firestore](https://firebase.google.com/docs/firestore)
- [Next.js Documentation](https://nextjs.org/docs)
