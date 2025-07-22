import os
import sys
import uvicorn
from pathlib import Path

# Add API_and_Models directory to path so we can import the api module
sys.path.append(str(Path(__file__).parent / "API_and_Models"))

# Set environment variables if needed
os.environ["API_KEY"] = os.environ.get("API_KEY", "your-secret-api-key")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run("API_and_Models.FastAPI_Backend_Server:app", host="0.0.0.0", port=8000, reload=True) 