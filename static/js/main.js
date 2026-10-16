/**
 * main.js - Core UI Interactions, Audio Reader & Confetti Celebrations
 * AgriScan AI - Plant Disease Detection Using CNN
 */

// Toast Notification Manager
function showToast(message, type = 'info') {
  let toastContainer = document.getElementById('toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 99999;
      display: flex;
      flex-direction: column;
      gap: 10px;
    `;
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  const colors = {
    info: 'border: 1px solid #10b981; background: rgba(16, 26, 20, 0.95); color: #f0fdf4;',
    error: 'border: 1px solid #f43f5e; background: rgba(35, 15, 20, 0.95); color: #ffe4e6;',
    warning: 'border: 1px solid #f59e0b; background: rgba(35, 25, 10, 0.95); color: #fef3c7;'
  };

  toast.style.cssText = `
    padding: 12px 20px;
    border-radius: 10px;
    backdrop-filter: blur(12px);
    font-size: 0.9rem;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.3s ease;
    transform: translateY(20px);
    opacity: 0;
    ${colors[type] || colors.info}
  `;

  toast.innerHTML = `<span>🍃</span> <div>${message}</div>`;
  toastContainer.appendChild(toast);

  // Animate in
  setTimeout(() => {
    toast.style.transform = 'translateY(0)';
    toast.style.opacity = '1';
  }, 10);

  // Remove after 4s
  setTimeout(() => {
    toast.style.transform = 'translateY(10px)';
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Text-to-Speech Audio Guidance for Farmers
function speakText(text) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel(); // Stop any ongoing speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    
    // Pick English or regional voice if available
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
    if (englishVoice) utterance.voice = englishVoice;

    window.speechSynthesis.speak(utterance);
    showToast('Reading agronomic recommendations aloud...', 'info');
  } else {
    showToast('Text-to-speech audio is not supported in this browser.', 'warning');
  }
}

// Interactive Confetti Effect for Healthy Plant Diagnoses
function fireHealthyConfetti() {
  const count = 40;
  const colors = ['#10b981', '#34d399', '#6ee7b7', '#facc15', '#a7f3d0'];

  for (let i = 0; i < count; i++) {
    const confetti = document.createElement('div');
    confetti.style.cssText = `
      position: fixed;
      top: 20%;
      left: ${Math.random() * 80 + 10}%;
      width: ${Math.random() * 8 + 6}px;
      height: ${Math.random() * 12 + 8}px;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      border-radius: 2px;
      pointer-events: none;
      z-index: 999999;
      transform: rotate(${Math.random() * 360}deg);
      opacity: 1;
      transition: all ${Math.random() * 2 + 1.5}s cubic-bezier(0.25, 1, 0.5, 1);
    `;
    document.body.appendChild(confetti);

    setTimeout(() => {
      confetti.style.transform = `translate(${Math.random() * 160 - 80}px, ${Math.random() * 450 + 200}px) rotate(${Math.random() * 720}deg)`;
      confetti.style.opacity = '0';
    }, 20);

    setTimeout(() => confetti.remove(), 3500);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Mobile nav toggle or active links
  const currentPath = window.location.pathname;
  document.querySelectorAll('.nav-item a').forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });
});
