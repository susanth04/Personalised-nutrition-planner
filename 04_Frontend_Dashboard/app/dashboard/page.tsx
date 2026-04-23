// app/dashboard/page.tsx - Example protected dashboard page
'use client';

import { useAuth } from '@/lib/auth-context';
import { useLogout } from '@/lib/auth-hooks';
import { useRouter } from 'next/navigation';
import { LogOut, User } from 'lucide-react';

export default function DashboardPage() {
  const router = useRouter();
  const { user, userProfile, loading } = useAuth();
  const { logout } = useLogout();

  // Redirect to login if not authenticated
  if (!loading && !user) {
    router.push('/login');
    return null;
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-12 h-12 bg-blue-200 rounded-full animate-pulse mx-auto mb-4"></div>
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const handleLogout = async () => {
    try {
      await logout();
      router.push('/login');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">NutritionWin Dashboard</h1>
            <p className="text-gray-600 mt-1">Welcome, {userProfile?.firstName} {userProfile?.lastName}!</p>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition"
          >
            <LogOut className="w-5 h-5" />
            Logout
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* User Info Card */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center">
              <User className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {userProfile?.firstName} {userProfile?.lastName}
              </h2>
              <p className="text-gray-600">{userProfile?.email}</p>
              <p className="text-sm text-gray-500 mt-1">
                Role: <span className="font-medium capitalize">{userProfile?.role}</span>
              </p>
            </div>
          </div>
        </div>

        {/* Content Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Card 1 */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Nutrition Plans</h3>
            <p className="text-gray-600 mb-4">View and manage your personalized nutrition plans</p>
            <button className="text-blue-600 hover:text-blue-700 font-medium">Learn more →</button>
          </div>

          {/* Card 2 */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Health Metrics</h3>
            <p className="text-gray-600 mb-4">Track your health metrics and progress</p>
            <button className="text-blue-600 hover:text-blue-700 font-medium">Learn more →</button>
          </div>

          {/* Card 3 */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Settings</h3>
            <p className="text-gray-600 mb-4">Update your account and preferences</p>
            <button className="text-blue-600 hover:text-blue-700 font-medium">Learn more →</button>
          </div>
        </div>
      </main>
    </div>
  );
}
