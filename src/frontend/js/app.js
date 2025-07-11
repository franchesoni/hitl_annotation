
// =====================
// State & DOM Refs
// =====================
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const img = new window.Image();
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const overlay = document.getElementById('loading-overlay');
const idDiv = document.getElementById('image-ids');

let imageIds = [];
let currentIdx = 0;
let isLoading = false;

// =====================
// UI Helpers
// =====================
function setLoading(loading) {
  isLoading = loading;
  updateNavButtons();
  if (overlay) overlay.style.display = loading ? 'flex' : 'none';
}

function updateNavButtons() {
  if (prevBtn) prevBtn.disabled = isLoading || (currentIdx <= 0);
  if (nextBtn) nextBtn.disabled = isLoading || (currentIdx >= imageIds.length - 1);
}

function updateIdDisplay() {
  if (idDiv) {
    idDiv.innerHTML = `<b>Image ID [${currentIdx+1}/${imageIds.length}]:</b><br><div>${imageIds[currentIdx] || ''}</div>`;
  }
}

// =====================
// Image Logic
// =====================
function drawImageToCanvas() {
  // Get viewport size (minus a small margin)
  const maxWidth = window.innerWidth * 0.90;
  const maxHeight = window.innerHeight * 0.90;
  const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
  const drawWidth = img.width * scale;
  const drawHeight = img.height * scale;
  canvas.width = drawWidth;
  canvas.height = drawHeight;
  ctx.drawImage(img, 0, 0, drawWidth, drawHeight);
  setLoading(false);
}

img.onload = drawImageToCanvas;

function updateImage() {
  if (imageIds.length === 0) return;
  setLoading(true);
  img.src = '/api/sample?id=' + encodeURIComponent(imageIds[currentIdx]);
  updateIdDisplay();
}

// =====================
// Navigation
// =====================
function goToPrev() {
  if (isLoading) return;
  if (currentIdx > 0) {
    currentIdx--;
    updateImage();
    updateNavButtons();
  }
}

function goToNext() {
  if (isLoading) return;
  if (currentIdx < imageIds.length - 1) {
    currentIdx++;
    updateImage();
    updateNavButtons();
  }
}

// =====================
// Initialization
// =====================
function fetchAndInitImages() {
  fetch('/api/ids')
    .then(r => r.json())
    .then(ids => {
      imageIds = ids;
      currentIdx = 0;
      updateImage();
      updateNavButtons();
    });
}

document.addEventListener('DOMContentLoaded', () => {
  fetchAndInitImages();
  if (prevBtn) prevBtn.addEventListener('click', goToPrev);
  if (nextBtn) nextBtn.addEventListener('click', goToNext);
});
