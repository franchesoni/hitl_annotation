const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const img    = new Image();


img.onload = () => {
  // Get viewport size (minus a small margin)
  const maxWidth = window.innerWidth * 0.95;
  const maxHeight = window.innerHeight * 0.95;
  // Calculate scale to fit image within viewport
  const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
  const drawWidth = img.width * scale;
  const drawHeight = img.height * scale;
  canvas.width = drawWidth;
  canvas.height = drawHeight;
  ctx.drawImage(img, 0, 0, drawWidth, drawHeight);
};

img.src = '/api/sample';   // talks to backend route
