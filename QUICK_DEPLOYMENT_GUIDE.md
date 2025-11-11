# 🚀 DEPLOYMENT QUICK START

## ✅ Current Status

### Frontend (04_Frontend_Dashboard)
- ✅ Next.js build successful
- ✅ All config files updated
- ✅ .gitignore properly configured
- ⏳ Vercel deployment in progress...

**Issue**: Vercel is trying to upload 840MB+ (should only be ~10MB)
**Cause**: `.vercel` folder or other cached files

### Backend (03_Machine_Learning_Pipeline)
- ✅ FastAPI server ready
- ✅ Dependencies listed in requirements.txt
- ✅ render.yaml configured
- ✅ .gitignore updated to include essential models
- ⏰ Ready to deploy to Render

---

## 🎯 RECOMMENDED DEPLOYMENT ORDER

### 1️⃣ Deploy Backend First (Render)

**Why?** Frontend needs the backend API URL

**Steps:**
1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click "New +" → "Web Service"
4. Select your repository: `Personalised-nutrition-planner`
5. **Configure:**
   ```
   Name: nutrition-api
   Root Directory: 03_Machine_Learning_Pipeline
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python render_launcher.py
   Instance Type: Free
   ```
6. Click "Create Web Service"
7. Wait 3-5 minutes
8. Copy the URL: `https://nutrition-api-xxxx.onrender.com`

### 2️⃣ Update Frontend with Backend URL

**After backend is deployed:**

```bash
cd 04_Frontend_Dashboard
```

Add environment variable in Vercel:
- Go to Vercel Dashboard → Your Project → Settings → Environment Variables
- Add: `NEXT_PUBLIC_API_URL` = `https://your-render-url.onrender.com`
- Or use CLI: `vercel env add NEXT_PUBLIC_API_URL production`

### 3️⃣ Deploy Frontend (Vercel - Via Dashboard)

**Instead of CLI (to avoid upload issues):**

1. **Stop current vercel CLI** (Ctrl+C if still running)
2. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Production ready - frontend and backend"
   git push origin master
   ```
3. **Deploy via Vercel Dashboard:**
   - Go to [vercel.com](https://vercel.com/new)
   - Import `Personalised-nutrition-planner`
   - Root Directory: `04_Frontend_Dashboard`
   - Framework: Next.js (auto-detected)
   - Add environment variable: `NEXT_PUBLIC_API_URL`
   - Click Deploy

---

## 🐛 FIX VERCEL UPLOAD ISSUE

### Problem
Vercel CLI is uploading 840MB instead of ~10MB

### Solution
```bash
# Stop the current upload (Ctrl+C)

# Clean up Vercel cache
cd C:\RESEARCH-PROJECT
rm -rf .vercel

cd 04_Frontend_Dashboard
rm -rf .vercel
rm -rf .next

# Use Dashboard instead of CLI (recommended)
# OR try CLI again:
vercel --prod
```

---

## 📋 FINAL CHECKLIST

### Backend
- [ ] Deploy to Render
- [ ] Verify `/docs` endpoint works
- [ ] Copy the Render URL
- [ ] Test API endpoints

### Frontend  
- [ ] Stop current Vercel CLI upload
- [ ] Add backend URL to environment variables
- [ ] Deploy via Vercel Dashboard (recommended)
- [ ] OR clean .vercel folder and retry CLI
- [ ] Test frontend loads
- [ ] Verify API calls work

### Integration
- [ ] Update CORS in backend with frontend URL
- [ ] Redeploy backend if CORS changed
- [ ] Test full user flow
- [ ] Check browser console for errors

---

## 🎉 EXPECTED RESULTS

**Backend**: `https://nutrition-api-xxxx.onrender.com`
- Docs at: `/docs`
- Health check: `/health`

**Frontend**: `https://your-project.vercel.app`
- All pages load
- API integration works
- No CORS errors

---

## 💡 PRO TIPS

1. **Use Dashboard over CLI** for first deployment (more reliable)
2. **Deploy backend first** so you have the URL ready
3. **Clear caches** if uploads seem too large
4. **Check .gitignore** ensures node_modules is excluded
5. **Monitor build logs** in both dashboards for issues

---

**Next Action**: Deploy backend to Render NOW, then update frontend with the URL!
