# AI-as-a-Service: Code-Free Machine Learning Platform

A full-stack web application for training ML models without coding. Upload CSV datasets, let AI detect targets, train multiple models, compare performance, and download production-ready models.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ (with pip)
- Node.js 18+ (with npm)
- Virtual environment (recommended)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment (Windows):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   The backend will be available at: `http://localhost:8000`
   
   API docs available at: `http://localhost:8000/docs`

### Frontend Setup

1. **Open a new terminal and navigate to frontend:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

   The frontend will be available at: `http://localhost:5173` (or the port Vite assigns)

### Using the Application

1. **Upload a CSV dataset:**
   - Click "Upload CSV" in the header
   - Select your dataset file

2. **AI Assistance:**
   - **Detect Target:** Let AI identify the target column
   - **Auto-Select Model:** AI picks the best model after quick testing
   - **Auto-Everything:** AI handles detection, preprocessing, training, and reporting

3. **Manual Training:**
   - Select target column from dropdown
   - Choose a model from the catalog
   - Click "Train Selected Model" or "Train All Models"

4. **View Results:**
   - Check the leaderboard for ranked models
   - View metrics, confusion matrices, ROC curves, and residuals
   - Download models (.pkl or .h5 format)
   - Download detailed PDF reports

5. **Make Predictions:**
   - Click "Predict" on any model in the leaderboard
   - Enter feature values in the modal
   - View predictions instantly

## 📁 Project Structure

```
Major/
├── backend/
│   ├── app/
│   │   ├── ml/              # DSA-style model implementations
│   │   ├── routers/         # API endpoints
│   │   ├── services/        # Business logic
│   │   ├── utils/           # Utilities (preprocessing, metrics, etc.)
│   │   └── main.py          # FastAPI app entry point
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── types.ts         # TypeScript types
│   │   └── style.css        # Styling
│   └── package.json
└── README.md
```

## 🎯 Features

- ✅ **No-Code ML Training:** Upload CSV and train models without writing code
- ✅ **AI Target Detection:** Automatic target column identification
- ✅ **Model Comparison:** Train all models and see leaderboard rankings
- ✅ **Detailed Reports:** PDF reports with metrics, confusion matrices, and visualizations
- ✅ **Model Downloads:** Download trained models in .pkl or .h5 format
- ✅ **Interactive Predictions:** Make predictions with a user-friendly form
- ✅ **Feature Hints:** Categorical value mappings shown during prediction
- ✅ **Loading States:** Visual feedback during all operations
- ✅ **Smooth Animations:** Polished UI with transitions

## 🔧 Available Models

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- SVM (Support Vector Machine)
- KNN (K-Nearest Neighbors)
- Gradient Boosting
- XGBoost (optional)

## 📝 API Endpoints

- `POST /upload` - Upload CSV dataset
- `POST /detect-target` - AI target detection
- `POST /auto-select` - AI model selection
- `POST /auto-everything` - Full AI workflow
- `POST /train` - Train single model
- `POST /train-all` - Train all models
- `POST /predict` - Make predictions
- `GET /models` - List saved models
- `GET /models/catalog` - List available models
- `GET /models/download/{id}` - Download .pkl model
- `GET /models/download/{id}/h5` - Download .h5 model
- `GET /report/download/{id}` - Download PDF report

## 🧹 Clearing Storage

To clear all stored models, datasets, reports, and artifacts (useful for testing):

```bash
cd backend
python clear_storage.py
```

This will:
- Delete all uploaded datasets
- Delete all trained models (.pkl and .h5 files)
- Delete all generated PDF reports
- Delete all plot images
- Reset all index files

**Note:** This action cannot be undone. Use with caution!

## 🛠️ Troubleshooting

**Backend won't start:**
- Ensure Python 3.10+ is installed
- Activate virtual environment
- Install all dependencies: `pip install -r requirements.txt`
- Check if port 8000 is available

**Frontend won't start:**
- Ensure Node.js 18+ is installed
- Run `npm install` in frontend directory
- Check if port 5173 (or assigned port) is available

**Models not training:**
- Ensure dataset has valid data
- Check target column is selected
- Verify CSV format is correct

**PDF reports empty:**
- Train models first (reports generated after "Train All Models")
- Check backend storage directory for generated reports

## 📦 Storage

All generated files are stored in:
- **Backend:** `backend/app/storage/`
  - `datasets/` - Uploaded CSV files
  - `models/` - Trained model files (.pkl, .h5)
  - `reports/` - Generated PDF reports
  - `artifacts/` - Plot images and visualizations

## 🎨 Development

**Backend Development:**
```bash
cd backend
.venv\Scripts\activate
uvicorn app.main:app --reload
```

**Frontend Development:**
```bash
cd frontend
npm run dev
```

**Build Frontend for Production:**
```bash
cd frontend
npm run build
npm run preview
```

## 📄 License

This project is provided as-is for educational and development purposes.

