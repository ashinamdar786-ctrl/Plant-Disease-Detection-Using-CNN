"""
test_app.py - Comprehensive Unit & Pipeline Verification Suite
Tests database persistence, MobileNetV2 CNN inference, Grad-CAM attention maps,
model evaluation metrics, Flask endpoints, input validation, and scan history.
"""

import os
import io
from PIL import Image
import app
import database
import model_engine
import disease_data

print("=" * 65)
print("1. TESTING DATABASE INITIALIZATION & STATS")
print("=" * 65)
database.init_db()
stats = database.get_stats()
print("Initial Database Stats:", stats)
assert isinstance(stats, dict), "Stats should return a dictionary"

print("\n" + "=" * 65)
print("2. TESTING CNN MODEL LOADING & GENUINE INFERENCE")
print("=" * 65)
test_sample = os.path.join('static', 'samples', 'tomato_early_blight.jpg')
assert os.path.exists(test_sample), f"Sample image {test_sample} must exist."

pred = app.cnn_model.predict(test_sample)
print("Predicted Crop:", pred['crop'])
print("Predicted Disease:", pred['disease'])
print(f"Top-1 Probability / Confidence: {pred['confidence']}%")
print("Top-3 Candidate Predictions:")
for rank, cand in enumerate(pred['top3'], start=1):
    print(f"  #{rank}: {cand['crop']} - {cand['disease']} ({cand['probability']}%)")

assert pred['confidence'] > 50.0, "Expected high confidence on clear early blight sample"
assert len(pred['top3']) == 3, "Expected exactly top 3 candidates"
assert pred['top3'][0]['probability'] >= pred['top3'][1]['probability'], "Top 1 prob must be >= Top 2"

print("\n" + "=" * 65)
print("3. TESTING GENUINE GRAD-CAM HEATMAP GENERATION")
print("=" * 65)
os.makedirs(os.path.join('static', 'heatmaps'), exist_ok=True)
heatmap_test = os.path.join('static', 'heatmaps', 'test_gradcam.jpg')
app.cnn_model.generate_gradcam_heatmap(pred, heatmap_test)
assert os.path.exists(heatmap_test), "Grad-CAM output heatmap file must be generated"
assert os.path.getsize(heatmap_test) > 0, "Grad-CAM output heatmap file must not be empty"
print("Grad-CAM overlay saved successfully:", heatmap_test, f"({os.path.getsize(heatmap_test)} bytes)")

print("\n" + "=" * 65)
print("4. TESTING GENUINE MODEL EVALUATION METRICS")
print("=" * 65)
metrics = model_engine.get_model_evaluation_metrics()
print(f"Overall Test Accuracy: {metrics['overall_accuracy']}%")
print(f"Macro Precision: {metrics['macro_precision']}%")
print(f"Macro Recall: {metrics['macro_recall']}%")
print(f"Macro F1-Score: {metrics['macro_f1']}%")
print(f"Total Test Samples: {metrics['total_test_samples']}")
print(f"Per-Class Count: {len(metrics['per_class_metrics'])}")
assert metrics['overall_accuracy'] >= 99.0, "Verified accuracy should be ~99.17%"
assert metrics['total_test_samples'] == 5430, "Verified test samples should be 5,430"

print("\n" + "=" * 65)
print("5. TESTING FLASK CLIENT ENDPOINTS & INPUT VALIDATION")
print("=" * 65)
client = app.app.test_client()

routes = ['/', '/diagnose', '/history', '/metrics', '/guide']
for route in routes:
    res = client.get(route)
    print(f"GET {route} -> Status: {res.status_code}")
    assert res.status_code == 200, f"Route {route} failed with status {res.status_code}"

# Test valid sample prediction via API
post_res = client.post('/api/predict', data={'sample_name': 'tomato_early_blight.jpg', 'notes': 'MCA Test Plot'})
print("POST /api/predict (Sample) -> Status:", post_res.status_code)
assert post_res.status_code == 200
data = post_res.get_json()
assert data['success'] is True
pred_id = data['prediction_id']
print(f"  Success: {data['success']}, Disease: {data['crop']} - {data['disease']}, Confidence: {data['confidence']}%")

# Test prescription report
report_res = client.get(f"/report/{pred_id}")
print(f"GET /report/{pred_id} -> Status: {report_res.status_code}")
assert report_res.status_code == 200

# Test valid file upload
img_buffer = io.BytesIO()
test_pil = Image.open(test_sample)
test_pil.save(img_buffer, format='JPEG')
img_buffer.seek(0)

upload_res = client.post('/api/predict', data={
    'image': (img_buffer, 'test_upload.jpg'),
    'notes': 'Uploaded File Test'
}, content_type='multipart/form-data')
print("POST /api/predict (Valid JPG Upload) -> Status:", upload_res.status_code)
assert upload_res.status_code == 200
upload_data = upload_res.get_json()
assert upload_data['success'] is True
print(f"  Upload Success: True, Confidence: {upload_data['confidence']}%")

# Test invalid file format validation (e.g. .txt file)
txt_buffer = io.BytesIO(b"This is not an image.")
invalid_res = client.post('/api/predict', data={
    'image': (txt_buffer, 'malicious.txt')
}, content_type='multipart/form-data')
print("POST /api/predict (Invalid Extension .txt) -> Status:", invalid_res.status_code)
assert invalid_res.status_code == 400
print("  Expected Error Message:", invalid_res.get_json().get('error'))

# Test corrupt image validation
corrupt_buffer = io.BytesIO(b"Corrupt data pretending to be jpg")
corrupt_res = client.post('/api/predict', data={
    'image': (corrupt_buffer, 'corrupt.jpg')
}, content_type='multipart/form-data')
print("POST /api/predict (Corrupt JPG Data) -> Status:", corrupt_res.status_code)
assert corrupt_res.status_code == 400
print("  Expected Error Message:", corrupt_res.get_json().get('error'))

print("\n" + "=" * 65)
print("6. TESTING SCAN HISTORY PERSISTENCE")
print("=" * 65)
history_records = database.get_all_predictions(limit=5)
print(f"Retrieved {len(history_records)} history records from database.")
assert len(history_records) >= 2, "Should have saved at least 2 predictions"
print(f"Latest record: #{history_records[0]['id']} {history_records[0]['crop']} - {history_records[0]['disease']} ({history_records[0]['confidence']}%)")

print("\n" + "=" * 65)
print("ALL VERIFICATION SUITE TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)")
print("=" * 65)
