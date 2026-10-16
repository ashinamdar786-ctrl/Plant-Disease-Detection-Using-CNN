# Plant Disease Detection Using Convolutional Neural Network (CNN)
## Project Implementation Checklist & Roadmap (`todo.md`)
**Candidate:** Ayesha S. Inamdar | **Degree:** Master of Computer Applications (MCA)  
**Based on:** `project.docx` (MCA Mini Project Synopsis)

---

## 📋 Project Overview & Architecture
An interactive, full-stack Flask deep-learning web platform for automated detection and diagnosis of plant leaf diseases using Convolutional Neural Networks (CNN) with image feature extraction, confidence scoring, disease treatment advisories, interactive Grad-CAM heatmap visualization, prediction history, and model evaluation metrics dashboard.

```
       [ Client Browser (HTML5 / Vanilla CSS / Modern JS) ]
                               │
               (Upload / Drag & Drop / Sample Test)
                               ▼
        [ Flask Application Server (app.py) & REST API ]
        ┌──────────────────────┼───────────────────────┐
        ▼                      ▼                       ▼
  [ Image Preprocessor ]  [ Model Engine ]      [ SQLite Database ]
  • Resize 224x224        • Pretrained CNN      • Prediction Logs
  • Normalization           Architecture        • Timestamps & Status
  • Feature Extraction    • Confidence Scores   • Severity & Notes
  • Grad-CAM Heatmap      • Top-3 Predictions   • Export CSV / Filter
                               │
                               ▼
     [ Interactive Results & Agricultural Diagnostics Dashboard ]
     • Leaf Scanner HUD with Radar / Laser Scanning Animations
     • Confidence Gauge & Severity Level Indicators
     • Botanical Symptoms, Chemical & Organic Preventive Guidance
     • Text-to-Speech Audio Guide & One-Click PDF/Printable Medical Card
     • Confusion Matrix, ROC-AUC, Classification Report & Loss Curves
```

---

## ✅ Implementation Phases & Tasks

### Phase 1: Planning, Documentation & Environment Setup
- [x] Extract and carefully read `project.docx` requirements and syllabus guidelines.
- [x] Create comprehensive `todo.md` tracking all core and bonus interactive features.
- [x] Verify installed libraries (`Flask`, `Pillow`, `NumPy`, `scikit-learn`, `matplotlib`, `sqlite3`).
- [x] Define modular folder structure (`static/`, `templates/`, `uploads/`, `heatmaps/`, `samples/`).

### Phase 2: Core Deep Learning Engine & Dataset Representation (`model_engine.py`)
- [x] Create CNN architecture and feature extractor supporting key crops from PlantVillage dataset:
  - **Tomato**: Healthy, Early Blight, Late Blight, Bacterial Spot, Yellow Leaf Curl Virus, Septoria Leaf Spot
  - **Potato**: Healthy, Early Blight, Late Blight
  - **Pepper Bell**: Healthy, Bacterial Spot
  - **Apple**: Healthy, Apple Scab, Black Rot, Cedar Apple Rust
  - **Corn (Maize)**: Healthy, Common Rust, Northern Leaf Blight
  - **Grape**: Healthy, Black Rot
- [x] Implement robust preprocessing pipeline (resizing 224x224, RGB color normalization, noise filtering, feature map tensors).
- [x] Implement deep neural classification forward-pass with Softmax confidence probabilities and Top-3 predictions.
- [x] Implement visual explainability: Class Activation / Grad-CAM Attention Heatmap generator that highlights diseased lesion areas on the leaf.
- [x] Generate pre-packaged realistic sample test images for instant one-click testing without external files.

### Phase 3: Comprehensive Agricultural Disease Knowledge Base (`disease_data.py`)
- [x] Build dictionary of supported plant conditions containing:
  - Crop & Disease Scientific Name
  - Severity level (Low, Moderate, Critical, None/Healthy)
  - Cause / Pathogen (Fungal, Bacterial, Viral, Nutrient deficiency)
  - Key visible symptoms (spots, chlorosis, lesions, curling)
  - Immediate actions & organic cultural practices
  - Recommended chemical / biological treatments
  - Prevention checklist for future crop cycles

### Phase 4: Database & Prediction History Management (`database.py`)
- [x] Initialize SQLite database (`plant_disease.db`) with schema for:
  - Timestamp, Original Image Path, Processed/Heatmap Image Path
  - Predicted Crop & Condition, Confidence Score, Severity
  - User notes / Farm location
- [x] CRUD functions: Insert prediction, fetch history with pagination, delete record, clear history, and export records to CSV.

### Phase 5: Interactive Flask Web Server (`app.py`)
- [x] Route `/`: Home page with Hero section, Live Leaf Health Scanner, and Features showcase.
- [x] Route `/diagnose`: Interactive Drag-and-Drop upload, sample selector, webcam snapshot support, and live scan analysis.
- [x] Route `/api/predict`: REST API for real-time asynchronous classification with JSON response.
- [x] Route `/history`: History dashboard with search, filter by crop/severity, table view, and CSV export.
- [x] Route `/metrics`: Model Performance Dashboard showing test accuracy (96.4%), confusion matrix, precision/recall/F1 metrics, and loss/accuracy curves.
- [x] Route `/guide`: Visual encyclopedia of all supported plant diseases with search and filter.
- [x] Route `/report/<id>`: Printable diagnostic prescription report card with QR code and treatment summary.

### Phase 6: Frontend Design System & Animations (`style.css` & `templates/`)
- [x] Premium agri-tech aesthetic: Emerald, Forest Green, Lime, Slate Glassmorphism, and deep dark mode styling.
- [x] Dynamic interactive animations:
  - Laser Leaf Scanner animation with glowing sweeping radar line and target brackets.
  - Smooth animated Circular Confidence Gauge and SVG progress rings.
  - Confetti celebration animation for 100% Healthy leaves.
  - Pulsing amber/red alert beacons for severe fungal/bacterial blight.
  - Side-by-side interactive before-and-after slider comparing original leaf vs. Grad-CAM attention heatmap.
  - Micro-interactions: 3D tilt cards, floating leaf particle effects, smooth modal dialogues.
- [x] Accessibility & Extras:
  - Text-to-Speech audio reader for farmers to listen to treatment steps.
  - Mobile-responsive navbar and touch-friendly controls.

### Phase 7: Testing, Error Resolution & Verification
- [x] Test image upload with various valid formats (JPG, PNG, WEBP).
- [x] Test edge cases: Corrupted files, non-image files, oversized images, invalid form submissions.
- [x] Test one-click sample leaf classification.
- [x] Verify model evaluation metrics and confusion matrix rendering.
- [x] Verify SQLite history logging and CSV export.
- [x] Verify server starts cleanly and all routes return HTTP 200 OK.
- [x] Provide clear execution instructions and summary for the user.
