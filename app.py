from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import PyPDF2
import re
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and vectorizer
try:
    with open('resume_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)
    
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model = None
    tfidf = None

def cleanResume(resumeText):
    """Clean resume text for processing"""
    resumeText = re.sub(r'\b\w{1,2}\b', '', resumeText)  # remove short words
    resumeText = re.sub(r'[^a-zA-Z]', ' ', resumeText)  # remove numbers and special characters
    return resumeText.lower()

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes"""
    try:
        text = ''
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def calculate_ats_score(job_description, resume_text):
    """Calculate ATS score based on keyword matching"""
    # Clean and tokenize job description
    job_keywords = set(re.findall(r'\b\w+\b', job_description.lower()))
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'a', 'an'}
    job_keywords = job_keywords - stop_words
    
    # Clean and tokenize resume
    resume_keywords = set(re.findall(r'\b\w+\b', resume_text.lower()))
    resume_keywords = resume_keywords - stop_words
    
    if len(job_keywords) == 0:
        return 0
    
    # Find matching keywords
    matched_keywords = job_keywords.intersection(resume_keywords)
    ats_score = len(matched_keywords) / len(job_keywords) * 100
    
    return min(ats_score, 100)  # Cap at 100%

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'ATS Scoring API is running!',
        'endpoints': {
            'health': '/health',
            'analyze_resume': '/analyze-resume (POST)',
            'calculate_ats_only': '/calculate-ats-only (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': tfidf is not None
    })

@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    """Analyze resume and calculate ATS score"""
    try:
        # Check if model is loaded
        if model is None or tfidf is None:
            return jsonify({
                'error': 'Model not loaded properly'
            }), 500
        
        # Get job description from request
        job_description = request.form.get('job_description', '')
        if not job_description:
            return jsonify({
                'error': 'Job description is required'
            }), 400
        
        # Get resume file
        if 'resume_file' not in request.files:
            return jsonify({
                'error': 'Resume file is required'
            }), 400
        
        resume_file = request.files['resume_file']
        if resume_file.filename == '':
            return jsonify({
                'error': 'No file selected'
            }), 400
        
        # Extract text from PDF
        pdf_bytes = resume_file.read()
        resume_text = extract_text_from_pdf_bytes(pdf_bytes)
        
        if not resume_text.strip():
            return jsonify({
                'error': 'Could not extract text from PDF'
            }), 400
        
        # Clean resume text
        cleaned_resume = cleanResume(resume_text)
        
        # Predict category
        vectorized_resume = tfidf.transform([cleaned_resume])
        prediction = model.predict(vectorized_resume)
        prediction_proba = model.predict_proba(vectorized_resume)
        
        # Calculate ATS score
        ats_score = calculate_ats_score(job_description, cleaned_resume)
        
        # Get confidence score
        confidence = max(prediction_proba[0]) * 100
        
        return jsonify({
            'success': True,
            'predicted_category': prediction[0],
            'confidence': round(confidence, 2),
            'ats_score': round(ats_score, 2),
            'resume_text_length': len(resume_text),
            'cleaned_text_length': len(cleaned_resume)
        })
        
    except Exception as e:
        print(f"Error in analyze_resume: {e}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/calculate-ats-only', methods=['POST'])
def calculate_ats_only():
    """Calculate ATS score only (for text inputs)"""
    try:
        data = request.get_json()
        
        job_description = data.get('job_description', '')
        resume_text = data.get('resume_text', '')
        
        if not job_description or not resume_text:
            return jsonify({
                'error': 'Both job_description and resume_text are required'
            }), 400
        
        # Clean resume text
        cleaned_resume = cleanResume(resume_text)
        
        # Calculate ATS score
        ats_score = calculate_ats_score(job_description, cleaned_resume)
        
        return jsonify({
            'success': True,
            'ats_score': round(ats_score, 2)
        })
        
    except Exception as e:
        print(f"Error in calculate_ats_only: {e}")
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
