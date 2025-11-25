# ATS Scoring API Setup

This repository contains a Flask API for ATS (Applicant Tracking System) scoring using machine learning.

## Files Included

- `app.py` - Main Flask API application
- `requirements.txt` - Python dependencies
- `train_model.py` - Script to train and generate model files
- `test_api.py` - Script to test the API
- `README.md` - This setup guide

## Step 1: Generate Model Files

First, you need to create the model files (`resume_classifier.pkl` and `tfidf_vectorizer.pkl`):

### Option A: Using Original Dataset
1. Download the resume dataset from [Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
2. Place the `UpdatedResumeDataSet.csv` file in the same directory
3. Run the training script:
```bash
python train_model.py
```

### Option B: Using Sample Data (if you don't have the dataset)
Simply run the training script and it will use sample data:
```bash
python train_model.py
```

This will create:
- `resume_classifier.pkl` (trained model)
- `tfidf_vectorizer.pkl` (text vectorizer)

## Step 2: Test Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Test the API:
```bash
python test_api.py
```

Or test manually:
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test ATS calculation
curl -X POST http://localhost:5000/calculate-ats-only \
  -H "Content-Type: application/json" \
  -d '{"job_description":"Python developer with Django experience","resume_text":"Experienced Python developer with Django and Flask"}'
```

## Step 3: Deploy to railway (Free)

1. Create a GitHub repository with all files:
   - app.py
   - requirements.txt
   - resume_classifier.pkl
   - tfidf_vectorizer.pkl

2. Go to [railway.com](https://render.com)](https://railway.com/) and sign up

3. Create a new Web Service:
   - Connect your GitHub repository
   - Use these settings:
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Instance Type**: Free

4. Deploy and get your API URL (e.g., `https://your-app.onrender.com`)

## Step 4: Test Deployed API

Replace `localhost:5000` with your deployed URL in test_api.py and run:
```bash
python test_api.py
```

## API Endpoints

### GET /health
Check if the API is running and models are loaded.

### POST /analyze-resume
Analyze a PDF resume and calculate ATS score.
- Form data: `job_description` (text), `resume_file` (PDF file)
- Returns: ATS score, predicted category, confidence

### POST /calculate-ats-only
Calculate ATS score from text inputs.
- JSON: `{"job_description": "...", "resume_text": "..."}`
- Returns: ATS score

## Integration with Your App

Update your frontend hooks to use the deployed API URL:

```typescript
const response = await fetch('https://your-deployed-api.onrender.com/analyze-resume', {
  method: 'POST',
  body: formData,
});
```

## Troubleshooting

1. **Model files too large**: The pickle files should be under 100MB for most free hosting services.

2. **Memory issues**: If you encounter memory issues, try reducing the `max_features` in TfidfVectorizer or use a smaller dataset.

3. **PDF extraction issues**: Make sure your PDF files are text-based (not scanned images).

## File Structure
```
ats-api/
├── app.py
├── requirements.txt
├── train_model.py
├── test_api.py
├── README.md
├── resume_classifier.pkl (generated)
└── tfidf_vectorizer.pkl (generated)
```
