"""
app.py - Main Flask Application Server
Plant Disease Detection Using Convolutional Neural Network (CNN)
Author: Ayesha S. Inamdar (MCA Mini Project)
"""

import os
import time
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, Response
from werkzeug.utils import secure_filename
from disease_data import DISEASES_DB, get_disease_info, get_all_classes
from model_engine import PlantCNNModel, init_sample_images, get_model_evaluation_metrics
import database

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'plant-cnn-agro-secret-key-2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
app.config['HEATMAP_FOLDER'] = os.path.join(app.static_folder, 'heatmaps')
app.config['SAMPLES_FOLDER'] = os.path.join(app.static_folder, 'samples')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['HEATMAP_FOLDER'], app.config['SAMPLES_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Initialize Database and DL Model
database.init_db()
cnn_model = PlantCNNModel()
samples_manifest = init_sample_images(app.config['SAMPLES_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home landing page with interactive hero, scanning demo, and synopsis details."""
    stats = database.get_stats()
    return render_template('index.html', stats=stats, samples=samples_manifest[:4])


@app.route('/diagnose')
def diagnose():
    """Interactive Leaf Diagnostic Studio with Drag & Drop, Samples, and Live HUD."""
    return render_template('diagnose.html', samples=samples_manifest, classes=get_all_classes())


from PIL import Image

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Asynchronous REST endpoint for CNN plant leaf prediction.
    Accepts JPG, JPEG, PNG, or WEBP file upload or sample leaf name.
    Performs genuine MobileNetV2 inference and Grad-CAM saliency extraction.
    """
    image_path = None
    original_filename = "sample.jpg"
    is_sample = False
    
    # Check if a sample preset was selected
    sample_name = request.form.get('sample_name')
    if sample_name:
        sample_file = os.path.join(app.config['SAMPLES_FOLDER'], sample_name)
        if os.path.exists(sample_file):
            image_path = sample_file
            original_filename = sample_name
            is_sample = True

    # Check if a file was uploaded
    if not image_path:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file uploaded. Please select an image.'}), 400
        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected. Please choose a file.'}), 400
        if not allowed_file(file.filename):
            return jsonify({
                'success': False, 
                'error': 'Unsupported file format. Please upload a valid JPG, JPEG, PNG, or WEBP image.'
            }), 400
        
        # Validate image integrity
        try:
            with Image.open(file) as test_img:
                test_img.verify()
            file.seek(0)
        except Exception:
            return jsonify({
                'success': False, 
                'error': 'Corrupt or unreadable image file. Please provide a clear, valid leaf photo.'
            }), 400
        
        filename = secure_filename(file.filename)
        unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{filename}"
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(saved_path)
        image_path = saved_path
        original_filename = filename

    notes = request.form.get('notes', '').strip()

    try:
        # Run genuine CNN model inference (no hardcoded or fake scores)
        result = cnn_model.predict(image_path)
        
        # Generate authentic Grad-CAM attention heatmap overlay
        heatmap_filename = f"heatmap_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        heatmap_full_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
        cnn_model.generate_gradcam_heatmap(result, heatmap_full_path)
        
        # Store relative paths for static rendering
        rel_img_path = f"/static/samples/{original_filename}" if is_sample else f"/static/uploads/{os.path.basename(image_path)}"
        rel_heatmap_path = f"/static/heatmaps/{heatmap_filename}"
        
        # Save to SQLite prediction history
        pred_id = database.add_prediction(
            filename=original_filename,
            image_path=rel_img_path,
            heatmap_path=rel_heatmap_path,
            crop=result["crop"],
            disease=result["disease"],
            confidence=result["confidence"],
            severity=result["severity"],
            is_healthy=result["is_healthy"],
            notes=notes
        )
        
        return jsonify({
            'success': True,
            'prediction_id': pred_id,
            'crop': result['crop'],
            'disease': result['disease'],
            'confidence': result['confidence'],
            'severity': result['severity'],
            'is_healthy': result['is_healthy'],
            'top3': result['top3'],
            'disease_info': result['disease_info'],
            'is_low_confidence': result.get('is_low_confidence', False),
            'warning': result.get('warning'),
            'image_url': rel_img_path,
            'heatmap_url': rel_heatmap_path,
            'report_url': url_for('view_report', pred_id=pred_id)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Image processing error: {str(e)}'}), 500


@app.route('/history')
def history():
    """Prediction History dashboard with search, filtering, and export."""
    crop_filter = request.args.get('crop', 'all')
    severity_filter = request.args.get('severity', 'all')
    search = request.args.get('search', '').strip()
    
    records = database.get_all_predictions(
        crop_filter=crop_filter,
        severity_filter=severity_filter,
        search=search if search else None
    )
    stats = database.get_stats()
    return render_template('history.html', records=records, stats=stats, crop_filter=crop_filter, severity_filter=severity_filter, search=search)


@app.route('/api/history/delete/<int:pred_id>', methods=['POST'])
def delete_history_item(pred_id):
    """Delete a single prediction from history."""
    success = database.delete_prediction(pred_id)
    return jsonify({'success': success})


@app.route('/api/history/clear', methods=['POST'])
def clear_all_history():
    """Clear all records from prediction history."""
    count = database.clear_history()
    return jsonify({'success': True, 'deleted': count})


@app.route('/api/export-csv')
def export_csv():
    """Export prediction logs as a downloadable CSV."""
    csv_data = database.export_predictions_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=plant_disease_predictions.csv"}
    )


@app.route('/metrics')
def metrics():
    """Model Performance Evaluation Dashboard."""
    eval_metrics = get_model_evaluation_metrics()
    return render_template('metrics.html', metrics=eval_metrics)


@app.route('/guide')
def guide():
    """Botanical Disease Encyclopedia & Treatment Handbook."""
    crop_filter = request.args.get('crop', 'all')
    all_diseases = DISEASES_DB
    if crop_filter != 'all':
        filtered = {k: v for k, v in all_diseases.items() if v['crop'].lower() == crop_filter.lower()}
    else:
        filtered = all_diseases
    return render_template('guide.html', diseases=filtered, selected_crop=crop_filter)


@app.route('/report/<int:pred_id>')
def view_report(pred_id):
    """Printable agronomic prescription card & diagnostic report."""
    record = database.get_prediction_by_id(pred_id)
    if not record:
        return redirect(url_for('diagnose'))
    
    # Match disease information
    matched_class = None
    for k, v in DISEASES_DB.items():
        if v['crop'] == record['crop'] and v['disease'] == record['disease']:
            matched_class = k
            break
            
    disease_info = get_disease_info(matched_class if matched_class else f"{record['crop']}___{record['disease']}")
    return render_template('report.html', record=record, disease_info=disease_info)


if __name__ == '__main__':
    print("Plant Disease Detection CNN Application starting on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)
