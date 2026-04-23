// Firestore User Data Service
import {
  collection,
  doc,
  setDoc,
  getDoc,
  updateDoc,
  deleteDoc,
  query,
  where,
  getDocs,
  Timestamp,
  DocumentData,
  QueryConstraint,
} from 'firebase/firestore';
import { db } from './firebase';

const USERS_COLLECTION = 'users';

export interface UserProfile {
  uid: string;
  email: string;
  firstName: string;
  lastName: string;
  role: string;
  createdAt: Date | Timestamp;
  updatedAt: Date | Timestamp;
  [key: string]: any; // Allow additional custom fields
}

/**
 * Store user data in Firestore
 */
export const setUserData = async (
  uid: string,
  userData: Partial<UserProfile>
): Promise<void> => {
  try {
    const userRef = doc(db, USERS_COLLECTION, uid);
    const dataToStore = {
      ...userData,
      uid,
      updatedAt: Timestamp.now(),
    };

    // Set or merge the document
    await setDoc(userRef, dataToStore, { merge: true });
  } catch (error) {
    console.error('Error storing user data:', error);
    throw new Error('Failed to store user data');
  }
};

/**
 * Retrieve user data from Firestore
 */
export const getUserData = async (uid: string): Promise<UserProfile | null> => {
  try {
    const userRef = doc(db, USERS_COLLECTION, uid);
    const docSnap = await getDoc(userRef);

    if (docSnap.exists()) {
      const data = docSnap.data();
      return {
        ...data,
        createdAt:
          data.createdAt instanceof Timestamp
            ? data.createdAt.toDate()
            : data.createdAt,
        updatedAt:
          data.updatedAt instanceof Timestamp
            ? data.updatedAt.toDate()
            : data.updatedAt,
      } as UserProfile;
    }

    return null;
  } catch (error) {
    console.error('Error retrieving user data:', error);
    throw new Error('Failed to retrieve user data');
  }
};

/**
 * Update user data in Firestore
 */
export const updateUserData = async (
  uid: string,
  userData: Partial<UserProfile>
): Promise<void> => {
  try {
    const userRef = doc(db, USERS_COLLECTION, uid);
    const dataToUpdate = {
      ...userData,
      updatedAt: Timestamp.now(),
    };

    await updateDoc(userRef, dataToUpdate);
  } catch (error) {
    console.error('Error updating user data:', error);
    throw new Error('Failed to update user data');
  }
};

/**
 * Delete user data from Firestore
 */
export const deleteUserData = async (uid: string): Promise<void> => {
  try {
    const userRef = doc(db, USERS_COLLECTION, uid);
    await deleteDoc(userRef);
  } catch (error) {
    console.error('Error deleting user data:', error);
    throw new Error('Failed to delete user data');
  }
};

/**
 * Query users by email
 */
export const getUserByEmail = async (email: string): Promise<UserProfile | null> => {
  try {
    const usersRef = collection(db, USERS_COLLECTION);
    const q = query(usersRef, where('email', '==', email));
    const querySnapshot = await getDocs(q);

    if (!querySnapshot.empty) {
      const data = querySnapshot.docs[0].data();
      return {
        ...data,
        createdAt:
          data.createdAt instanceof Timestamp
            ? data.createdAt.toDate()
            : data.createdAt,
        updatedAt:
          data.updatedAt instanceof Timestamp
            ? data.updatedAt.toDate()
            : data.updatedAt,
      } as UserProfile;
    }

    return null;
  } catch (error) {
    console.error('Error querying user by email:', error);
    throw new Error('Failed to query user');
  }
};

/**
 * Query users by role
 */
export const getUsersByRole = async (role: string): Promise<UserProfile[]> => {
  try {
    const usersRef = collection(db, USERS_COLLECTION);
    const q = query(usersRef, where('role', '==', role));
    const querySnapshot = await getDocs(q);

    return querySnapshot.docs.map((doc) => {
      const data = doc.data();
      return {
        ...data,
        createdAt:
          data.createdAt instanceof Timestamp
            ? data.createdAt.toDate()
            : data.createdAt,
        updatedAt:
          data.updatedAt instanceof Timestamp
            ? data.updatedAt.toDate()
            : data.updatedAt,
      } as UserProfile;
    });
  } catch (error) {
    console.error('Error querying users by role:', error);
    throw new Error('Failed to query users');
  }
};

/**
 * Get all users
 */
export const getAllUsers = async (): Promise<UserProfile[]> => {
  try {
    const usersRef = collection(db, USERS_COLLECTION);
    const querySnapshot = await getDocs(usersRef);

    return querySnapshot.docs.map((doc) => {
      const data = doc.data();
      return {
        ...data,
        createdAt:
          data.createdAt instanceof Timestamp
            ? data.createdAt.toDate()
            : data.createdAt,
        updatedAt:
          data.updatedAt instanceof Timestamp
            ? data.updatedAt.toDate()
            : data.updatedAt,
      } as UserProfile;
    });
  } catch (error) {
    console.error('Error retrieving all users:', error);
    throw new Error('Failed to retrieve users');
  }
};

/**
 * Check if user exists
 */
export const userExists = async (uid: string): Promise<boolean> => {
  try {
    const userRef = doc(db, USERS_COLLECTION, uid);
    const docSnap = await getDoc(userRef);
    return docSnap.exists();
  } catch (error) {
    console.error('Error checking user existence:', error);
    return false;
  }
};
