// app/register/page.tsx
import RegisterComponent from '@/app/components/RegisterComponent';

export const metadata = {
  title: 'Register - NutritionWin',
  description: 'Create a new NutritionWin account',
};

export default function RegisterPage() {
  return <RegisterComponent />;
}
