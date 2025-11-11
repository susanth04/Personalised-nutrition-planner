@echo off
echo Microbiome Health Platform - Deployment Setup
echo ==============================================

echo.
echo This script will help you deploy the application to Vercel and Render.
echo Make sure you have:
echo 1. Vercel CLI installed (npm i -g vercel)
echo 2. Render account and CLI (if available)
echo 3. GitHub repository set up
echo.

set /p DEPLOY_FRONTEND="Deploy frontend to Vercel? (y/n): "
if /i "%DEPLOY_FRONTEND%"=="y" (
    echo.
    echo Deploying frontend to Vercel...
    echo Make sure you're in the frontend directory and logged in to Vercel
    echo.
    cd 04_Frontend_Dashboard
    vercel --prod
    cd ..
)

echo.
set /p DEPLOY_BACKEND="Deploy backend to Render? (y/n): "
if /i "%DEPLOY_BACKEND%"=="y" (
    echo.
    echo Backend deployment instructions:
    echo 1. Go to render.com and create a new Web Service
    echo 2. Connect your GitHub repository
    echo 3. Set the following configuration:
    echo    - Runtime: Python 3
    echo    - Build Command: pip install -r requirements.txt
    echo    - Start Command: python render_launcher.py
    echo    - Root Directory: 03_Machine_Learning_Pipeline
    echo 4. Add environment variables from .env.example
    echo.
    echo Press any key when ready to continue...
    pause > nul
)

echo.
echo Deployment setup complete!
echo.
echo Next steps:
echo 1. Update the CORS origins in FastAPI backend with your Vercel domain
echo 2. Update NEXT_PUBLIC_API_URL in Vercel with your Render backend URL
echo 3. Test the deployed application
echo.
echo Check DEPLOYMENT_README.md for detailed instructions.

pause