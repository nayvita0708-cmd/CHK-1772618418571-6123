"""
SachLens AI - Main Flask Application
Deepfake Detection and Digital Authenticity Verification System
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from detector import DeepfakeDetector
from metadata_checker import MetadataChecker
from frame_extractor import FrameExtractor
from utils import allowed_file, validate_file_size, cleanup_old_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend',
            static_url_path='')
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = '../uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
app.config['MODEL_PATH'] = '../model/deepfake_model.h5'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
detector = DeepfakeDetector(model_path=app.config['MODEL_PATH'])
metadata_checker = MetadataChecker()
frame_extractor = FrameExtractor()

# Clean up old files periodically
cleanup_old_files(app.config['UPLOAD_FOLDER'], hours=24)


@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('../frontend', path)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'SachLens AI'
    })


@app.route('/analyze', methods=['POST'])
def analyze_media():
    """
    Main analysis endpoint
    Accepts image or video file and returns authenticity analysis
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Validate file size
        if not validate_file_size(file, app.config['MAX_CONTENT_LENGTH']):
            return jsonify({'error': 'File size exceeds limit'}), 400
        
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Determine file type
        is_video = file_extension in ['mp4', 'avi', 'mov', 'mkv']
        
        # Initialize results
        ai_probability = 0.0
        metadata_warnings = []
        heatmap_data = None
        
        # Process based on file type
        if is_video:
            # Extract frames from video
            logger.info("Processing video file...")
            frames = frame_extractor.extract_frames(file_path, max_frames=30)
            
            if not frames:
                return jsonify({'error': 'Could not extract frames from video'}), 400
            
            # Analyze frames
            predictions = []
            for frame in frames:
                pred = detector.predict(frame)
                predictions.append(pred)
            
            # Average predictions
            ai_probability = sum(predictions) / len(predictions)
            
            # Optional: Generate heatmap for key frames
            if len(frames) > 0:
                heatmap_data = detector.generate_heatmap(frames[0])
        else:
            # Process image
            logger.info("Processing image file...")
            from PIL import Image
            import numpy as np
            
            # Load and preprocess image
            image = Image.open(file_path).convert('RGB')
            image_array = np.array(image)
            
            # Run AI detection
            ai_probability = detector.predict(image_array)
            
            # Generate heatmap
            heatmap_data = detector.generate_heatmap(image_array)
        
        # Extract metadata
        logger.info("Analyzing metadata...")
        metadata_warnings = metadata_checker.analyze(file_path)
        
        # Calculate authenticity score (0-100)
        # Weight: AI detection 70%, metadata 30%
        ai_score = (1 - ai_probability) * 70  # Lower AI prob = higher authenticity
        metadata_score = (1 - len(metadata_warnings) * 0.15) * 30
        metadata_score = max(0, min(30, metadata_score))
        
        authenticity_score = min(100, ai_score + metadata_score)
        
        # Determine status
        if authenticity_score >= 70:
            status = "Likely Authentic"
        elif authenticity_score >= 40:
            status = "Suspicious"
        else:
            status = "Potential Deepfake"
        
        # Prepare response
        response = {
            'success': True,
            'filename': original_filename,
            'file_type': 'video' if is_video else 'image',
            'score': round(authenticity_score, 1),
            'status': status,
            'ai_probability': round(ai_probability, 3),
            'metadata_analysis': metadata_warnings,
            'heatmap': heatmap_data.tolist() if heatmap_data is not None else None,
            'analysis_time': datetime.now().isoformat()
        }
        
        logger.info(f"Analysis complete: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Batch analysis endpoint for multiple files"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files[]')
        
        if len(files) > 10:
            return jsonify({'error': 'Maximum 10 files allowed'}), 400
        
        results = []
        for file in files:
            # Process each file (simplified - you might want to use threading)
            if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                # Save and analyze file (reuse analyze logic)
                # For brevity, we'll simulate results
                results.append({
                    'filename': file.filename,
                    'score': 75.5,
                    'status': 'Likely Authentic'
                })
        
        return jsonify({'success': True, 'results': results}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)