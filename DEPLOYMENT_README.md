# Microbiome Health Platform - Deployment Guide

## Overview
This project consists of a Next.js frontend deployed on Vercel and a FastAPI backend deployed on Render.

## Frontend Deployment (Vercel)

### Prerequisites
- Vercel account
- GitHub repository

### Deployment Steps
1. Connect your GitHub repository to Vercel
2. Configure build settings:
   - Framework Preset: Next.js
   - Root Directory: `04_Frontend_Dashboard`
   - Build Command: `npm run build`
   - Output Directory: `.next`

3. Add environment variables in Vercel dashboard:
   - `NEXT_PUBLIC_API_URL`: Your Render backend URL (e.g., `https://your-app.onrender.com`)

4. Deploy

## Backend Deployment (Render)

### Prerequisites
- Render account
- GitHub repository

### Deployment Steps
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Configure settings:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python render_launcher.py`
   - Root Directory: `03_Machine_Learning_Pipeline`

4. Add environment variables:
   - `API_KEY`: Your API key
   - `PORT`: 8000 (automatically set by Render)

5. Deploy

## Post-Deployment Configuration

1. Update CORS in `API_and_Models/FastAPI_Backend_Server.py`:
   ```python
   allow_origins=[
       "https://your-frontend-domain.vercel.app",  # Your actual Vercel domain
   ]
   ```

2. Update the Vercel environment variable `NEXT_PUBLIC_API_URL` with your Render URL

3. Test the deployment by visiting your Vercel frontend URL

## File Structure
```
C:\RESEARCH-PROJECT\
├── 04_Frontend_Dashboard\     # Next.js frontend
│   ├── vercel.json           # Vercel configuration
│   ├── .env.example          # Environment variables template
│   └── ...                   # Next.js app files
└── 03_Machine_Learning_Pipeline\  # FastAPI backend
    ├── requirements.txt      # Python dependencies
    ├── render.yaml          # Render configuration
    ├── Procfile             # Alternative deployment config
    ├── runtime.txt          # Python version
    ├── render_launcher.py   # Production launcher
    ├── .env.example         # Environment variables template
    └── API_and_Models\      # Backend code and models
```

## Troubleshooting

### Frontend Issues
- Check Vercel build logs for Next.js errors
- Verify environment variables are set correctly
- Ensure API URL is accessible

### Backend Issues
- Check Render logs for Python errors
- Verify all model files are present in `API_and_Models/`
- Ensure dependencies are compatible with Python 3.9

### CORS Issues
- Update allowed origins in FastAPI CORS middleware
- Ensure frontend domain is added to allowed origins
- Check that API URL environment variable is correct