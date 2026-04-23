// Firebase Configuration
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getAnalytics } from 'firebase/analytics';

const firebaseConfig = {
  apiKey: "AIzaSyBRqumqQoWU5FdnuN_WTn4O0tjiCy1YZSc",
  authDomain: "nutritwin-cd71b.firebaseapp.com",
  projectId: "nutritwin-cd71b",
  storageBucket: "nutritwin-cd71b.firebasestorage.app",
  messagingSenderId: "1040958981519",
  appId: "1:1040958981519:web:e2df547f97ab68ab64b7fa",
  measurementId: "G-F5SKB6R34D"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Initialize Cloud Firestore and get a reference to the service
export const db = getFirestore(app);

// Initialize Analytics (only in browser)
let analytics;
if (typeof window !== 'undefined') {
  analytics = getAnalytics(app);
}

export default app;
