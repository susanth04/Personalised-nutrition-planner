# 🚀 Quick Start - Frontend Deployment

## Fastest Way to Deploy (5 minutes)

### 1️⃣ Deploy to Vercel
```bash
# Option 1: Via GitHub (Recommended)
1. Push your code to GitHub
2. Go to https://vercel.com/new
3. Import your repository
4. Set root directory: `04_Frontend_Dashboard`
5. Add environment variable: NEXT_PUBLIC_API_URL=https://your-api-url.com
6. Click Deploy
```

### 2️⃣ Or Deploy via CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend folder
cd 04_Frontend_Dashboard

# Install dependencies
npm install

# Deploy
vercel

# For production
vercel --prod
```

## Environment Variables to Set

In Vercel Dashboard → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL = https://your-backend-api.com
```

## Done! ✅

Your app will be live at: `https://your-project.vercel.app`

---

## Before You Deploy - Quick Check

✅ Run local build test:
```bash
npm install
npm run build
npm start
```

✅ Update API URL in `vercel.json` line 10:
```json
"destination": "https://your-actual-backend-api.com/api/:path*"
```

✅ Commit and push all changes:
```bash
git add .
git commit -m "Ready for deployment"
git push
```

## Common Issues

**Problem**: Build fails
**Solution**: 
```bash
rm -rf .next node_modules package-lock.json
npm install
npm run build
```

**Problem**: API calls fail
**Solution**: Check CORS settings on your backend API to allow your Vercel domain

**Problem**: Environment variables not working
**Solution**: Make sure they start with `NEXT_PUBLIC_` and redeploy after adding them

---

For detailed instructions, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)
