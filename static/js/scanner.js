/**
 * scanner.js - Leaf Diagnostic Studio & Interactive Grad-CAM Slider Controller
 * AgriScan AI - Plant Disease Detection Using CNN
 */

document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('leaf-file-input');
  const dropZone = document.getElementById('upload-dropzone');
  const sampleChips = document.querySelectorAll('.sample-chip');
  const diagnoseForm = document.getElementById('diagnose-form');
  const selectedSampleInput = document.getElementById('selected-sample-input');
  
  // UI Panels
  const previewContainer = document.getElementById('upload-preview-box');
  const previewImg = document.getElementById('preview-image');
  const scanningHud = document.getElementById('scanning-hud');
  const scanningStatusText = document.getElementById('scanning-status-text');
  const resultCard = document.getElementById('diagnostic-result-card');
  const initialPlaceholder = document.getElementById('initial-placeholder');

  let currentFile = null;
  let currentSample = null;

  // Drag and drop handlers
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
      }, false);
    });

    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    dropZone.addEventListener('click', () => {
      fileInput.click();
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    const validExtensions = ['jpg', 'jpeg', 'png', 'webp'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(ext) || !file.type.match('image.*')) {
      showToast('Invalid file format. Please upload a JPG, JPEG, PNG, or WEBP image file.', 'error');
      return;
    }
    currentFile = file;
    currentSample = null;
    selectedSampleInput.value = '';

    // Deselect sample chips
    sampleChips.forEach(c => c.classList.remove('active'));

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewContainer.style.display = 'block';
      showToast('Leaf image loaded. Ready to scan!', 'info');
      startScanAnalysis();
    };
    reader.readAsDataURL(file);
  }

  // Sample picker chips
  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const sampleName = chip.getAttribute('data-sample');
      sampleChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      currentSample = sampleName;
      currentFile = null;
      selectedSampleInput.value = sampleName;

      const sampleImg = chip.querySelector('img').src;
      previewImg.src = sampleImg;
      previewContainer.style.display = 'block';

      showToast(`Selected sample: ${chip.querySelector('span').innerText}`, 'info');
      startScanAnalysis();
    });
  });

  // Execute Scan Analysis
  function startScanAnalysis() {
    if (!currentFile && !currentSample) {
      showToast('Please select or upload a leaf first', 'warning');
      return;
    }

    // Show Scanning HUD
    initialPlaceholder.style.display = 'none';
    resultCard.style.display = 'none';
    scanningHud.style.display = 'block';

    const steps = [
      'Normalizing RGB pixel tensor (224 x 224)...',
      'Passing through Conv2D feature extraction layers...',
      'Evaluating disease pattern probability distributions...',
      'Computing Grad-CAM attention heatmap overlay...'
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      stepIdx = (stepIdx + 1) % steps.length;
      if (scanningStatusText) scanningStatusText.innerText = steps[stepIdx];
    }, 450);

    const formData = new FormData(diagnoseForm);
    if (currentFile) {
      formData.set('image', currentFile);
    }
    if (currentSample) {
      formData.set('sample_name', currentSample);
    }

    fetch('/api/predict', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      clearInterval(interval);
      setTimeout(() => {
        scanningHud.style.display = 'none';
        if (data.success) {
          renderDiagnosticResult(data);
        } else {
          showToast(data.error || 'Prediction failed', 'error');
          initialPlaceholder.style.display = 'block';
        }
      }, 700);
    })
    .catch(err => {
      clearInterval(interval);
      scanningHud.style.display = 'none';
      initialPlaceholder.style.display = 'block';
      showToast('Network error while analyzing leaf: ' + err.message, 'error');
    });
  }

  // Render Result & Dynamic Components
  function renderDiagnosticResult(data) {
    resultCard.style.display = 'block';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Set titles
    document.getElementById('res-crop-name').innerText = data.crop;
    document.getElementById('res-disease-name').innerText = data.disease;
    document.getElementById('res-sci-name').innerText = data.disease_info.scientific_name;

    // Severity & Status Badge
    const badge = document.getElementById('res-status-badge');
    badge.className = 'status-badge';
    if (data.is_healthy) {
      badge.classList.add('status-healthy');
      badge.innerHTML = `<span>●</span> Healthy Condition`;
      fireHealthyConfetti();
    } else if (data.severity === 'Critical') {
      badge.classList.add('status-critical');
      badge.innerHTML = `<span>⚠</span> Critical Severity`;
    } else {
      badge.classList.add('status-moderate');
      badge.innerHTML = `<span>●</span> Moderate Severity`;
    }

    // Low Confidence Banner handling
    const lowConfBanner = document.getElementById('low-confidence-banner');
    const lowConfText = document.getElementById('low-confidence-text');
    if (data.is_low_confidence && lowConfBanner && lowConfText) {
      lowConfText.innerText = data.warning || 'Confidence is below 45%. The leaf may be unsupported or lighting unclear.';
      lowConfBanner.style.display = 'block';
    } else if (lowConfBanner) {
      lowConfBanner.style.display = 'none';
    }

    // Confidence Circular SVG Gauge
    const gaugeFill = document.getElementById('gauge-circle-fill');
    const gaugeText = document.getElementById('gauge-text-val');
    gaugeText.innerText = `${data.confidence}%`;
    const circumference = 2 * Math.PI * 32; // r=32
    const offset = circumference - (data.confidence / 100) * circumference;
    gaugeFill.style.strokeDasharray = `${circumference} ${circumference}`;
    gaugeFill.style.strokeDashoffset = circumference;
    setTimeout(() => {
      gaugeFill.style.strokeDashoffset = offset;
    }, 100);

    // Setup Grad-CAM Slider
    const origSliderImg = document.getElementById('slider-orig-img');
    const heatSliderImg = document.getElementById('slider-heat-img');
    origSliderImg.src = data.image_url;
    heatSliderImg.src = data.heatmap_url;
    initGradCAMSlider();

    // Top-3 Candidates
    const top3Container = document.getElementById('top3-candidates-list');
    top3Container.innerHTML = '';
    data.top3.forEach((cand, idx) => {
      const row = document.createElement('div');
      row.style.cssText = 'margin-bottom: 10px; font-size: 0.85rem;';
      row.innerHTML = `
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span><strong>#${idx + 1}</strong> ${cand.crop} - ${cand.disease}</span>
          <span style="font-family: var(--font-mono); color: var(--emerald-400);">${cand.probability}%</span>
        </div>
        <div style="height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden;">
          <div style="height: 100%; width: ${cand.probability}%; background: ${idx === 0 ? 'var(--emerald-400)' : 'rgba(16,185,129,0.4)'}; border-radius: 99px; transition: width 0.8s ease;"></div>
        </div>
      `;
      top3Container.appendChild(row);
    });

    // Populate Symptoms & Treatments
    populateList('symptoms-list', data.disease_info.symptoms);
    populateList('organic-list', data.disease_info.organic_treatment);
    populateList('chemical-list', data.disease_info.chemical_treatment);
    populateList('prevention-list', data.disease_info.prevention);
    document.getElementById('cause-text').innerText = data.disease_info.cause;

    // Report Link
    const reportBtn = document.getElementById('view-report-btn');
    if (reportBtn) reportBtn.href = data.report_url;

    // Audio Speech reader setup
    const audioBtn = document.getElementById('speech-audio-btn');
    if (audioBtn) {
      audioBtn.onclick = () => {
        const speechMsg = `Diagnosis report for ${data.crop}: Condition is ${data.disease} with ${data.confidence} percent confidence. Cause: ${data.disease_info.cause}. Recommended organic treatment: ${data.disease_info.organic_treatment.join('. ')}`;
        speakText(speechMsg);
      };
    }
  }

  function populateList(elementId, items) {
    const list = document.getElementById(elementId);
    if (!list) return;
    list.innerHTML = '';
    items.forEach(item => {
      const li = document.createElement('li');
      li.className = 'advisory-item';
      li.innerHTML = `<span>✓</span> <div>${item}</div>`;
      list.appendChild(li);
    });
  }

  // Grad-CAM Interactive Comparison Slider
  function initGradCAMSlider() {
    const container = document.getElementById('gradcam-slider');
    const overlay = document.getElementById('slider-overlay');
    const handle = document.getElementById('slider-handle');
    if (!container || !handle || !overlay) return;

    let isDragging = false;

    function setSliderPosition(x) {
      const rect = container.getBoundingClientRect();
      let pos = (x - rect.left) / rect.width;
      pos = Math.max(0.05, Math.min(0.95, pos));
      const percentage = pos * 100;
      overlay.style.width = `${percentage}%`;
      handle.style.left = `${percentage}%`;
    }

    handle.onmousedown = () => isDragging = true;
    window.onmouseup = () => isDragging = false;
    window.onmousemove = (e) => {
      if (!isDragging) return;
      setSliderPosition(e.clientX);
    };

    // Touch events for mobile/tablets
    handle.ontouchstart = () => isDragging = true;
    window.ontouchend = () => isDragging = false;
    window.ontouchmove = (e) => {
      if (!isDragging || !e.touches[0]) return;
      setSliderPosition(e.touches[0].clientX);
    };

    // Reset to 50%
    overlay.style.width = '50%';
    handle.style.left = '50%';
  }

  // Tab Switching
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabTarget = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.tab-content').forEach(c => {
        c.classList.remove('active');
      });
      const activeContent = document.getElementById(tabTarget);
      if (activeContent) activeContent.classList.add('active');
    });
  });

  // Re-scan trigger
  const rescanBtn = document.getElementById('rescan-btn');
  if (rescanBtn) {
    rescanBtn.addEventListener('click', () => {
      startScanAnalysis();
    });
  }
});
