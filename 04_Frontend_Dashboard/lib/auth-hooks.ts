// Custom React hooks for Firebase Authentication
'use client';

import { useState } from 'react';
import { User } from 'firebase/auth';
import {
  registerUser,
  loginUser,
  logoutUser,
  resetPassword,
  getAuthErrorMessage,
} from './auth-service';

export interface UseRegisterOptions {
  onSuccess?: (user: User) => void;
  onError?: (error: Error) => void;
}

export interface UseLoginOptions {
  onSuccess?: (user: User) => void;
  onError?: (error: Error) => void;
}

export interface UsePasswordResetOptions {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

/**
 * Hook for user registration
 */
export const useRegister = (options?: UseRegisterOptions) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const register = async (
    email: string,
    password: string,
    firstName: string,
    lastName: string,
    role?: string
  ) => {
    setLoading(true);
    setError(null);

    try {
      const user = await registerUser(email, password, firstName, lastName, role);
      options?.onSuccess?.(user);
      return user;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Registration failed';
      setError(errorMessage);
      options?.onError?.(new Error(errorMessage));
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { register, loading, error };
};

/**
 * Hook for user login
 */
export const useLogin = (options?: UseLoginOptions) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = async (email: string, password: string) => {
    setLoading(true);
    setError(null);

    try {
      const user = await loginUser(email, password);
      options?.onSuccess?.(user);
      return user;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      options?.onError?.(new Error(errorMessage));
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { login, loading, error };
};

/**
 * Hook for user logout
 */
export const useLogout = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const logout = async () => {
    setLoading(true);
    setError(null);

    try {
      await logoutUser();
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Logout failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { logout, loading, error };
};

/**
 * Hook for password reset
 */
export const usePasswordReset = (options?: UsePasswordResetOptions) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const reset = async (email: string) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      await resetPassword(email);
      setSuccess(true);
      options?.onSuccess?.();
      return true;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Password reset failed';
      setError(errorMessage);
      options?.onError?.(new Error(errorMessage));
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { reset, loading, error, success };
};
