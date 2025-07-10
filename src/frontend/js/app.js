const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');


const img    = new window.Image();
let isLoading = false;

function setLoading(loading) {
  isLoading = loading;
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  const overlay = document.getElementById('loading-overlay');
  if (prevBtn) prevBtn.disabled = loading || (currentIdx <= 0);
  if (nextBtn) nextBtn.disabled = loading || (currentIdx >= imageIds.length - 1);
  if (overlay) overlay.style.display = loading ? 'flex' : 'none';
}

function drawImageToCanvas() {
  // Get viewport size (minus a small margin)
  const maxWidth = window.innerWidth * 0.90;
  const maxHeight = window.innerHeight * 0.90;
  // Calculate scale to fit image within viewport
  const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
  const drawWidth = img.width * scale;
  const drawHeight = img.height * scale;
  canvas.width = drawWidth;
  canvas.height = drawHeight;
  ctx.drawImage(img, 0, 0, drawWidth, drawHeight);
  setLoading(false);
}

img.onload = drawImageToCanvas;



let imageIds = [];
let currentIdx = 0;

function updateImage() {
  if (imageIds.length === 0) return;
  setLoading(true);
  img.src = '/api/sample?id=' + encodeURIComponent(imageIds[currentIdx]);
  updateIdDisplay();
}

function updateIdDisplay() {
  const idDiv = document.getElementById('image-ids');
  if (idDiv) {
    idDiv.innerHTML = `<b>Image ID [${currentIdx+1}/${imageIds.length}]:</b><br><div>${imageIds[currentIdx] || ''}</div>`;
  }
}


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

function updateNavButtons() {
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  if (prevBtn) prevBtn.disabled = (currentIdx <= 0);
  if (nextBtn) nextBtn.disabled = (currentIdx >= imageIds.length - 1);
}

document.addEventListener('DOMContentLoaded', () => {
  fetchAndInitImages();
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (isLoading) return;
      if (currentIdx > 0) {
        currentIdx--;
        updateImage();
        updateNavButtons();
      }
    });
  }
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      if (isLoading) return;
      if (currentIdx < imageIds.length - 1) {
        currentIdx++;
        updateImage();
        updateNavButtons();
      }
    });
  }
});
