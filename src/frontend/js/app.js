
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
// Zoom & Pan State
let scale = 1;
let panX = 0, panY = 0;
let minScale = 1;
const MAX_SCALE = 10;
let dragging = false, startX = 0, startY = 0;
let panZoomEnabled = true;


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
        idDiv.innerHTML = `<b>Image ID [${currentIdx + 1}/${imageIds.length}]:</b><br><div>${imageIds[currentIdx] || ''}</div>`;
    }
}

// =====================
// Image Logic
// =====================
function drawImageToCanvas() {
    // Set canvas to viewport size (fixed)
    const maxWidth = window.innerWidth * 0.90;
    const maxHeight = window.innerHeight * 0.90;
    canvas.width = maxWidth;
    canvas.height = maxHeight;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(scale, 0, 0, scale, panX, panY);
    ctx.drawImage(img, 0, 0, img.width, img.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0); // reset
    setLoading(false);
}

// Helper to reset view for a new image
function resetViewToImage() {
    const maxWidth = window.innerWidth * 0.90;
    const maxHeight = window.innerHeight * 0.90;
    const fitScale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
    scale = fitScale;
    minScale = fitScale * 0.5;
    panX = (maxWidth - img.width * scale) / 2;
    panY = (maxHeight - img.height * scale) / 2;
}
img.onload = function () {
    resetViewToImage();
    drawImageToCanvas();
};

function updateImage() {
    if (imageIds.length === 0) return;
    setLoading(true);
    img.src = '/api/sample?id=' + encodeURIComponent(imageIds[currentIdx]);
    updateIdDisplay();
}

// =====================
// Navigation
// =====================
function fetchImageList() {
    return fetch('/api/ids')
        .then(r => r.json())
        .then(ids => {
            const prevId = imageIds[currentIdx];
            imageIds = ids;
            // Find the index of the previous image in the new list
            let idx = imageIds.indexOf(prevId);
            if (idx === -1) {
                idx = 0;
            }
            currentIdx = idx;
            updateIdDisplay();
            updateNavButtons();
        });
}

function goToPrev() {
    if (isLoading) return;
    if (currentIdx > 0) {
        currentIdx--;
        updateImage();
        updateNavButtons();
        // Fetch the image list after navigating
        fetchImageList();
    }
}

function goToNext() {
    if (isLoading) return;
    if (currentIdx < imageIds.length - 1) {
        currentIdx++;
        updateImage();
        updateNavButtons();
        // Fetch the image list after navigating
        fetchImageList();
    }
}

// =====================
// Initialization
// =====================
document.addEventListener('DOMContentLoaded', () => {
    fetchImageList().then(() => updateImage());
    if (prevBtn) prevBtn.addEventListener('click', goToPrev);
    if (nextBtn) nextBtn.addEventListener('click', goToNext);

    // --- Zoom and Pan ---
    // Mouse wheel zoom (zoom around cursor)
    canvas.addEventListener('wheel', (e) => {
        if (!panZoomEnabled) return;
        e.preventDefault();
        if (!img.width || !img.height) return;
        const rect = canvas.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        // Image coords under mouse before zoom
        const imgX = (px - panX) / scale;
        const imgY = (py - panY) / scale;
        let newScale = scale * Math.pow(1.2, -e.deltaY / 200);
        newScale = Math.max(minScale, Math.min(MAX_SCALE, newScale));
        // Update pan so (imgX, imgY) stays under cursor
        panX = px - imgX * newScale;
        panY = py - imgY * newScale;
        scale = newScale;
        drawImageToCanvas();
    }, { passive: false });

    // Drag to pan
    canvas.addEventListener('pointerdown', (e) => {
        if (!panZoomEnabled) return;
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
        e.preventDefault();
    });
    canvas.addEventListener('pointermove', (e) => {
        if (!panZoomEnabled) return;
        if (!dragging) return;
        panX += e.clientX - startX;
        panY += e.clientY - startY;
        startX = e.clientX;
        startY = e.clientY;
        drawImageToCanvas();
        e.preventDefault();
    });
    canvas.addEventListener('pointerup', (e) => {
        if (!panZoomEnabled) return;
        dragging = false;
        canvas.releasePointerCapture(e.pointerId);
    });
    window.addEventListener('pointerup', () => { dragging = false; });

    // Pan/Zoom toggle button
    const toggleBtn = document.getElementById('toggle-panzoom-btn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            panZoomEnabled = !panZoomEnabled;
            // Reset view on toggle
            resetViewToImage();
            drawImageToCanvas();
            toggleBtn.textContent = panZoomEnabled ? 'Disable Pan/Zoom' : 'Enable Pan/Zoom';
        });
        // Set initial label
        toggleBtn.textContent = panZoomEnabled ? 'Disable Pan/Zoom' : 'Enable Pan/Zoom';
    }

    // Use existing Reset View button
    const resetBtn = document.getElementById('reset-view-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            resetViewToImage();
            drawImageToCanvas();
        });
    }

    // Redraw on resize
    window.addEventListener('resize', () => {
        drawImageToCanvas();
    });
});
