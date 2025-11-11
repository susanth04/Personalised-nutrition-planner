# 🎨 Personalized Nutrition Platform - Frontend Dashboard

A modern, responsive Next.js dashboard for personalized nutrition planning powered by microbiome analysis and machine learning.

## 🌟 Features

- 🔐 **Authentication System** - Secure login/register with PHP backend
- 📊 **Interactive Dashboard** - Real-time visualization of nutrition data
- 🍽️ **Meal Planning** - AI-powered personalized meal recommendations
- 🧬 **Digital Twin Simulation** - Microbiome-based health predictions
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile
- 🎨 **Modern UI** - Built with shadcn/ui and Tailwind CSS

## 🛠️ Tech Stack

- **Framework**: [Next.js 15.2.4](https://nextjs.org/) (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui (Radix UI primitives)
- **Charts**: Recharts
- **State Management**: React Hooks
- **Form Handling**: React Hook Form + Zod
- **HTTP Client**: Axios
- **Deployment**: Vercel

## 📦 Project Structure

```
04_Frontend_Dashboard/
├── app/                      # Next.js app directory
│   ├── layout.tsx           # Root layout
│   ├── page.tsx             # Home/Dashboard page
│   ├── about/               # About page
│   ├── digital-twin/        # Digital twin simulation page
│   └── meal-plan/           # Meal planning page
├── components/              # Reusable UI components
│   └── ui/                  # shadcn/ui components
├── lib/                     # Utility functions
├── hooks/                   # Custom React hooks
├── styles/                  # Global styles
├── public/                  # Static assets
├── config/                  # Configuration files
├── next.config.mjs          # Next.js configuration
├── tailwind.config.ts       # Tailwind CSS configuration
├── tsconfig.json            # TypeScript configuration
└── vercel.json              # Vercel deployment config
```

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/pnpm
- Backend API running (default: http://localhost:8000)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/susanth04/Personalised-nutrition-planner.git
cd Personalised-nutrition-planner/04_Frontend_Dashboard
```

2. **Install dependencies**
```bash
npm install
# or
pnpm install
```

3. **Set up environment variables**
```bash
# Copy the example env file
cp .env.local.example .env.local

# Edit .env.local and set your API URL
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. **Run the development server**
```bash
npm run dev
```

5. **Open your browser**
Navigate to [http://localhost:3000](http://localhost:3000)

## 🔧 Available Scripts

```bash
npm run dev      # Start development server (port 3000)
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

## 🌐 Deployment

### Quick Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

1. Click the button above or go to [vercel.com](https://vercel.com)
2. Import your repository
3. Set root directory to `04_Frontend_Dashboard`
4. Add environment variable: `NEXT_PUBLIC_API_URL`
5. Deploy!

**Detailed instructions**: See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

**Quick start**: See [QUICKSTART.md](./QUICKSTART.md)

## 🔑 Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | Yes | `http://localhost:8000` |

**Note**: Variables prefixed with `NEXT_PUBLIC_` are exposed to the browser.

## 📱 Pages

### Dashboard (`/`)
Main dashboard showing:
- User health metrics
- Microbiome analysis overview
- Quick access to features

### Meal Plan (`/meal-plan`)
- Personalized meal recommendations
- Nutritional breakdown
- Dietary preferences management

### Digital Twin (`/digital-twin`)
- Microbiome simulation interface
- Health predictions
- Intervention analysis

### About (`/about`)
- Project information
- Research background
- Technology overview

## 🎨 UI Components

Built with [shadcn/ui](https://ui.shadcn.com/), includes:
- Accordion, Alert Dialog, Avatar
- Button, Card, Checkbox
- Dialog, Dropdown Menu, Form
- Input, Label, Select
- Tabs, Toast, Tooltip
- Charts (via Recharts)
- And many more...

All components are fully customizable and accessible.

## 🔐 Authentication

The frontend integrates with a PHP backend for authentication:
- `login.php` - User login
- `register.php` - User registration
- `logout.php` - Session termination
- `session.php` - Session validation

## 🌈 Styling

- **Tailwind CSS**: Utility-first CSS framework
- **CSS Variables**: For theming support
- **Dark Mode**: Built-in theme switching
- **Responsive**: Mobile-first design
- **Animations**: Tailwind CSS Animate

## 🧪 API Integration

The frontend communicates with the backend API using Axios:

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL;

// Example API call
const response = await axios.get(`${API_URL}/api/meal-plan`);
```

## 📊 Data Visualization

Uses Recharts for interactive charts:
- Line charts for trends
- Bar charts for comparisons
- Pie charts for distributions
- Area charts for compositions

## 🔒 Security Features

- XSS Protection enabled
- CSRF protection (via backend)
- Secure headers configured
- Content Security Policy ready
- No sensitive data in client code

## 🐛 Troubleshooting

### Common Issues

**Build Errors**
```bash
# Clear cache and reinstall
rm -rf .next node_modules package-lock.json
npm install
npm run build
```

**API Connection Issues**
- Check `NEXT_PUBLIC_API_URL` is set correctly
- Verify backend API is running
- Check CORS settings on backend

**TypeScript Errors**
- Currently set to ignore build errors (development mode)
- To enable strict checking, update `next.config.mjs`

## 📚 Documentation

- [Deployment Guide](./DEPLOYMENT_GUIDE.md) - Complete deployment instructions
- [Quick Start](./QUICKSTART.md) - 5-minute deployment guide
- [Next.js Docs](https://nextjs.org/docs) - Next.js documentation
- [Tailwind CSS](https://tailwindcss.com/docs) - Styling documentation
- [shadcn/ui](https://ui.shadcn.com/) - Component documentation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of a research initiative on personalized nutrition using microbiome analysis.

## 👥 Authors

- **Susanth** - [GitHub](https://github.com/susanth04)

## 🙏 Acknowledgments

- Research team for ML models and backend API
- shadcn for the amazing UI components
- Vercel for hosting and deployment platform
- Next.js team for the excellent framework

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review deployment guides

---

**Built with ❤️ using Next.js and modern web technologies**

**Last Updated**: November 11, 2025
