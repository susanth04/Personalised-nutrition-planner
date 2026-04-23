# Firebase Setup and Installation Guide

## Quick Start

### 1. Install Firebase SDK

```bash
npm install firebase
```

### 2. Verify Firebase Configuration

The Firebase configuration is already set up in [lib/firebase.ts](lib/firebase.ts) with your NutritionWin Firebase project:

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

### 3. Build and Run

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Visit `http://localhost:3000` to see the application.

## Available Pages

- **Login**: http://localhost:3000/login
- **Register**: http://localhost:3000/register
- **Forgot Password**: http://localhost:3000/forgot-password
- **Dashboard**: http://localhost:3000/dashboard (protected, requires login)

## Firebase Console Setup

### 1. Authentication Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select the **nutritwin-cd71b** project
3. Navigate to **Authentication** > **Sign-in method**
4. Enable **Email/Password** provider

### 2. Firestore Database Setup

1. In Firebase Console, go to **Firestore Database**
2. Create a database in production mode
3. Set security rules (see below)

### 3. Security Rules for Firestore

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

## File Organization

```
lib/
├── firebase.ts                  # Firebase initialization
├── auth-service.ts              # Authentication functions
├── firestore-service.ts         # User data management
├── auth-context.tsx             # React context provider
├── auth-hooks.ts                # Custom hooks
└── protected-route.tsx          # Protected route wrapper

app/
├── layout.tsx                   # Root layout with AuthProvider
├── login/page.tsx               # Login page
├── register/page.tsx            # Registration page
├── forgot-password/page.tsx     # Password reset page
├── dashboard/page.tsx           # Protected dashboard
└── components/
    ├── LoginComponent.tsx       # Login form component
    ├── RegisterComponent.tsx    # Registration form component
    └── ForgotPasswordComponent.tsx # Password reset form component
```

## Usage Examples

### Using Authentication Hooks

```typescript
'use client';

import { useLogin } from '@/lib/auth-hooks';
import { useRouter } from 'next/navigation';

export default function MyLoginPage() {
  const router = useRouter();
  const { login, loading, error } = useLogin({
    onSuccess: () => {
      router.push('/dashboard');
    },
  });

  const handleSubmit = async (email: string, password: string) => {
    try {
      await login(email, password);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    // Your login form here
  );
}
```

### Using Auth Context

```typescript
'use client';

import { useAuth } from '@/lib/auth-context';

export default function MyComponent() {
  const { user, userProfile, loading, isAuthenticated } = useAuth();

  if (loading) return <div>Loading...</div>;

  if (!isAuthenticated) {
    return <div>Please log in</div>;
  }

  return (
    <div>
      Welcome, {userProfile?.firstName}!
    </div>
  );
}
```

### Direct Service Usage

```typescript
import { 
  registerUser, 
  loginUser, 
  logoutUser, 
  resetPassword 
} from '@/lib/auth-service';
import { 
  setUserData, 
  getUserData, 
  updateUserData 
} from '@/lib/firestore-service';

// Registration
const user = await registerUser(
  'user@example.com',
  'password123',
  'John',
  'Doe',
  'user'
);

// Login
const user = await loginUser('user@example.com', 'password123');

// Password Reset
await resetPassword('user@example.com');

// Logout
await logoutUser();

// Firestore operations
const userProfile = await getUserData(user.uid);
await updateUserData(user.uid, { firstName: 'Jane' });
```

## Testing

### Test Registration Flow
1. Go to http://localhost:3000/register
2. Fill in the form with:
   - First Name: Test
   - Last Name: User
   - Email: test@example.com
   - Password: password123
   - Role: user
3. Click Register
4. Check Firebase Console under Firestore to verify user data was saved

### Test Login Flow
1. Go to http://localhost:3000/login
2. Enter email and password from registration
3. Click Login
4. Should be redirected to dashboard

### Test Password Reset
1. Go to http://localhost:3000/forgot-password
2. Enter registered email
3. Click "Send Reset Link"
4. Check email for Firebase password reset link (or check Firebase Console logs)

## Debugging

### Check Auth State
```typescript
import { getCurrentUser, setupAuthStateListener } from '@/lib/auth-service';

// Get current user immediately
const user = getCurrentUser();
console.log('Current user:', user);

// Listen for auth state changes
const unsubscribe = setupAuthStateListener((user) => {
  console.log('Auth state changed:', user);
});
```

### Check Firestore Data
```typescript
import { getUserData, getAllUsers } from '@/lib/firestore-service';

// Get specific user
const profile = await getUserData('user-uid');
console.log('User profile:', profile);

// Get all users
const allUsers = await getAllUsers();
console.log('All users:', allUsers);
```

### Browser Console
The application logs authentication state changes to the browser console. Open the browser DevTools (F12) to see detailed logs.

## Troubleshooting

### Issue: "Firebase is not defined"
**Solution**: Ensure Firebase SDK is installed:
```bash
npm install firebase
```

### Issue: "Auth/email-already-in-use"
**Solution**: The email is already registered. Use a different email or reset the database.

### Issue: "Permission denied" error in Firestore
**Solution**: Check Firestore security rules. They should allow users to access their own documents.

### Issue: Password reset email not received
**Solution**: 
1. Check Firebase Console for email verification
2. Check spam/junk email folder
3. Verify sender email is configured in Firebase

## Next Steps

1. Customize the login/register components to match your branding
2. Add additional user profile fields as needed
3. Implement role-based access control
4. Set up email verification (optional)
5. Add social authentication (Google, GitHub, etc.)

## Support

For Firebase issues, visit:
- [Firebase Documentation](https://firebase.google.com/docs)
- [Firebase Support](https://firebase.google.com/support)

For Next.js issues, visit:
- [Next.js Documentation](https://nextjs.org/docs)
