# Deployment Checklist for Microbiome Health Platform

## Pre-Deployment Preparation

### 1. Repository Setup
- [ ] Create GitHub repository
- [ ] Push code to GitHub
- [ ] Ensure .gitignore files are working

### 2. Frontend (Vercel) Preparation
- [ ] Test build: `cd 04_Frontend_Dashboard && npm run build`
- [ ] Verify vercel.json configuration
- [ ] Check that API calls use environment variables
- [ ] Ensure no large files are included

### 3. Backend (Render) Preparation
- [ ] Test locally: `cd 03_Machine_Learning_Pipeline && python render_launcher.py`
- [ ] Verify requirements.txt has all dependencies
- [ ] Check that model files are accessible
- [ ] Ensure render.yaml or Procfile is correct

## Vercel Deployment Steps

### Option 1: Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login to Vercel
vercel login

# Deploy frontend
cd 04_Frontend_Dashboard
vercel --prod
```

### Option 2: Vercel Dashboard
1. Go to vercel.com
2. Click "New Project"
3. Import GitHub repository
4. Configure:
   - Framework Preset: Next.js
   - Root Directory: `04_Frontend_Dashboard`
   - Build Command: `npm run build`
   - Install Command: `npm install`
5. Add environment variables:
   - `NEXT_PUBLIC_API_URL`: `https://your-render-app.onrender.com`

## Render Deployment Steps

### Option 1: Render Dashboard
1. Go to render.com
2. Click "New" → "Web Service"
3. Connect GitHub repository
4. Configure:
   - Name: `microbiome-api`
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python render_launcher.py`
   - Root Directory: `03_Machine_Learning_Pipeline`
5. Add environment variables:
   - `API_KEY`: `your-secret-api-key`
   - `PORT`: `8000` (auto-set by Render)

### Option 2: render.yaml (Blueprint)
1. Push render.yaml to repository
2. Go to render.com → Blueprints
3. Connect repository
4. Deploy blueprint

## Post-Deployment Configuration

### 1. Update CORS Settings
After getting your Vercel domain (e.g., `https://your-app.vercel.app`):

Edit `03_Machine_Learning_Pipeline/API_and_Models/FastAPI_Backend_Server.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://localhost:3000",
        "https://your-app.vercel.app",  # Add your actual Vercel domain
        "*"  # Remove this in production for security
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. Update Frontend API URL
In Vercel dashboard, update environment variable:
- `NEXT_PUBLIC_API_URL`: `https://your-render-service.onrender.com`

### 3. Test Deployment
1. Visit Vercel frontend URL
2. Check API status indicator
3. Test meal plan generation
4. Test digital twin simulation

## Troubleshooting

### Frontend Issues
- **Build fails**: Check Next.js build logs, fix TypeScript errors
- **API not connecting**: Verify NEXT_PUBLIC_API_URL environment variable
- **CORS errors**: Update allowed origins in backend

### Backend Issues
- **Import errors**: Check Python path in render_launcher.py
- **Model loading fails**: Verify model files are in API_and_Models/
- **Port issues**: Ensure PORT environment variable is used

### Performance Issues
- **Slow loading**: Enable Vercel Analytics
- **Memory issues**: Check Render service logs
- **Large files**: Ensure .gitignore is excluding unnecessary files

## Security Checklist

- [ ] Remove debug mode in production
- [ ] Use secure API keys
- [ ] Enable HTTPS (automatic on Vercel/Render)
- [ ] Restrict CORS origins
- [ ] Monitor for sensitive data exposure

## Monitoring and Maintenance

### Vercel
- Check deployment logs
- Monitor function execution times
- Use Vercel Analytics for performance

### Render
- Monitor service health
- Check logs for errors
- Scale resources as needed
- Set up alerts for downtime

## Cost Optimization

### Vercel (Free Tier)
- 100GB bandwidth/month
- 1000 functions/month
- Automatic scaling

### Render (Free Tier)
- 750 hours/month
- 512MB RAM
- 1GB disk space

For production workloads, consider paid plans based on usage.