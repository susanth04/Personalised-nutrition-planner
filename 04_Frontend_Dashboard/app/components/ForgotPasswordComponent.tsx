// Forgot Password Component
'use client';

import { useState } from 'react';
import { usePasswordReset } from '@/lib/auth-hooks';
import { Mail, AlertCircle, CheckCircle, ArrowLeft } from 'lucide-react';

export default function ForgotPasswordComponent() {
  const { reset, loading, error, success } = usePasswordReset({
    onSuccess: () => {
      // Reset form after success
      setEmail('');
    },
  });

  const [email, setEmail] = useState('');
  const [validationError, setValidationError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError('');

    // Basic validation
    if (!email) {
      setValidationError('Please enter your email address');
      return;
    }

    if (!email.includes('@')) {
      setValidationError('Please enter a valid email address');
      return;
    }

    try {
      await reset(email);
    } catch (err) {
      // Error is handled by the hook
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-amber-50 to-orange-100 px-4">
      <div className="w-full max-w-md bg-white rounded-lg shadow-lg p-8">
        {/* Back Button */}
        <a
          href="/login"
          className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-700 mb-6 text-sm font-medium"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Login
        </a>

        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">Reset Password</h1>
          <p className="text-gray-600">
            {success
              ? 'Check your email for password reset instructions'
              : 'Enter your email to receive password reset instructions'}
          </p>
        </div>

        {!success ? (
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email Input */}
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 text-gray-400 w-5 h-5" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@example.com"
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent outline-none transition"
                />
              </div>
            </div>

            {/* Error Messages */}
            {(error || validationError) && (
              <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0" />
                <p className="text-sm text-red-700">{error || validationError}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white font-semibold py-2 rounded-lg transition duration-200"
            >
              {loading ? 'Sending reset link...' : 'Send Reset Link'}
            </button>
          </form>
        ) : (
          <div className="space-y-6">
            {/* Success Message */}
            <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-lg">
              <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-green-800 mb-1">Email sent successfully!</p>
                <p className="text-sm text-green-700">
                  Please check your email inbox (and spam folder) for instructions to reset your
                  password. The link will expire in 24 hours.
                </p>
              </div>
            </div>

            {/* Instructions */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-900 mb-2">What's next?</h3>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>✓ Check your email for a reset link</li>
                <li>✓ Click the link to set a new password</li>
                <li>✓ Return to login with your new password</li>
              </ul>
            </div>

            {/* Return to Login Button */}
            <a
              href="/login"
              className="block w-full text-center bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition duration-200"
            >
              Back to Login
            </a>
          </div>
        )}

        {/* Help Text */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-xs text-gray-600 text-center">
            Didn't receive the email?{' '}
            <button
              onClick={() => {
                setValidationError('');
                setEmail('');
              }}
              className="text-blue-600 hover:text-blue-700 font-medium"
            >
              Try again
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
