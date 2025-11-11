# Microbiome Health Platform - Deployment File Structure

## Frontend (Vercel) - Only Essential Files
```
04_Frontend_Dashboard/
├── app/                    # Next.js app directory
├── components/             # React components
├── lib/                    # Utilities and API client
├── public/                 # Static assets
├── styles/                 # CSS styles
├── package.json            # Dependencies
├── next.config.mjs         # Next.js config
├── vercel.json            # Vercel deployment config
├── tailwind.config.ts      # Tailwind config
├── tsconfig.json          # TypeScript config
├── .gitignore             # Exclude large files
└── README.md              # Deployment instructions
```

## Backend (Render) - Only Essential Files
```
03_Machine_Learning_Pipeline/
├── API_and_Models/        # Core API and models
│   ├── FastAPI_Backend_Server.py    # Main API server
│   ├── XGBoost_Hyperparameter_Configuration.json  # Model config
│   ├── Feature_Scaler_Model.pkl     # Feature scaler
│   ├── XGBoost_Model_Predictions.csv # Model predictions
│   └── [essential model files only]
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── Procfile              # Alternative deployment
├── runtime.txt           # Python version
├── render_launcher.py    # Production launcher
├── .gitignore           # Exclude large files
└── .env.example         # Environment variables
```

## Excluded Files (Not Deployed)

### Large Data Files (All directories)
- `*.csv` - Large microbiome datasets
- `*.tsv` - Tab-separated data files
- `*.txt` - Large analysis reports
- `*.log` - Log files

### Research Directories (Not needed for production)
- `IHMP/` - Raw research data
- `AGORA*/` - Metabolic model databases
- `FRANZOSA/` - Reference datasets
- `01_Raw_Data_Processing/` - Data processing scripts
- `02_Reference_Datasets/` - Reference data
- `05_Research_Documentation/` - Research papers
- `06_Processed_Datasets/` - Processed data files

### Development Files
- `node_modules/` - Frontend dependencies (installed on Vercel)
- `__pycache__/` - Python cache files
- `.next/` - Next.js build output
- IDE files (`.vscode/`, `.idea/`)

### Model Files (Selective inclusion)
Keep only essential model files for production:
- ✅ `XGBoost_Hyperparameter_Configuration.json`
- ✅ `Feature_Scaler_Model.pkl`
- ✅ `XGBoost_Model_Predictions.csv`
- ❌ Large PyTorch models (`.pth`, `.h5`) if not needed
- ❌ Raw training data

## Deployment Size Optimization

### Frontend Bundle Size
- Next.js automatically optimizes bundles
- Large assets are handled by Vercel's CDN
- API calls fetch data from backend

### Backend Memory Usage
- Only essential model files loaded
- Large datasets not included (use external storage if needed)
- Python dependencies optimized for production

## File Size Guidelines

### Vercel (Frontend)
- Total deployment size: < 100MB
- Individual files: < 25MB
- Static assets optimized automatically

### Render (Backend)
- Total deployment size: < 500MB
- Memory limit: 512MB - 2GB depending on plan
- Model files: Keep under 200MB total

## Performance Optimizations

1. **Lazy Loading**: Large components loaded on demand
2. **API Caching**: Backend responses cached where appropriate
3. **CDN**: Static assets served via Vercel's CDN
4. **Compression**: Automatic gzip compression
5. **Model Optimization**: Use quantized models if possible