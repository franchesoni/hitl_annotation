async function saveCurrentImageClassAnnotation() {
    const imgId = imageIds[currentIdx];
    if (!imgId) return;
    const selectedClass = imageSelectedClass[imgId];
    if (selectedClass) {
        // Save or update annotation
        await fetch('/api/save_label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: imgId, class: selectedClass })
        });
    } else {
        // Delete annotation if unselected
        await fetch('/api/save_label', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: imgId })
        });
    }
}

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
const maxSideScale = 0.75; // Fraction of window size used for canvas and fitting
let dragging = false, startX = 0, startY = 0;
let panZoomEnabled = true;

// Global class list state
let globalClasses = []; // [class1, class2, ...]
// Per-image selected class
let imageSelectedClass = {}; // { imageId: className }


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
    updateClassListDisplay();
}

function updateClassListDisplay() {
    const classListDiv = document.getElementById('class-list-container');
    if (!classListDiv) return;
    if (globalClasses.length === 0) {
        classListDiv.innerHTML = '<b>Classes:</b> <i>None</i>';
        return;
    }
    const imgId = imageIds[currentIdx];
    const selected = imgId ? imageSelectedClass[imgId] : undefined;
    classListDiv.innerHTML = '<b>Classes:</b> ' + globalClasses.map(c => {
        const isSelected = c === selected;
        return `<button class="class-btn" data-class="${c}" style="display:inline-block;background:${isSelected ? '#b3e5fc' : '#eee'};border-radius:4px;padding:2px 8px;margin:2px;border:${isSelected ? '2px solid #0288d1' : '1px solid #ccc'};cursor:pointer;">${c}</button>`;
    }).join(' ');

    // Add click listeners for class selection
    Array.from(classListDiv.querySelectorAll('.class-btn')).forEach(btn => {
        btn.addEventListener('click', () => {
            if (!imgId) return;
            const className = btn.dataset.class;
            if (imageSelectedClass[imgId] === className) {
                // Unselect if already selected
                delete imageSelectedClass[imgId];
            } else {
                imageSelectedClass[imgId] = className;
            }
            updateClassListDisplay();
        });
    });
}

// =====================
// Image Logic
// =====================
function drawImageToCanvas() {
    // Set canvas to viewport size (fixed)
    const maxWidth = window.innerWidth * maxSideScale;
    const maxHeight = window.innerHeight * maxSideScale;
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
    const maxWidth = window.innerWidth * maxSideScale;
    const maxHeight = window.innerHeight * maxSideScale;
    const fitScale = Math.min(maxWidth / img.width, maxHeight / img.height);
    scale = fitScale;
    minScale = Math.min(0.5, fitScale * 0.5);
    maxScale = Math.max(20, fitScale * 20);
    panX = (maxWidth - img.width * scale) / 2;
    panY = (maxHeight - img.height * scale) / 2;
}
img.onload = function () {
    resetViewToImage();
    drawImageToCanvas();
};

async function updateImage() {
    if (imageIds.length === 0) return;
    setLoading(true);
    const imgId = imageIds[currentIdx];
    img.src = '/api/sample?id=' + encodeURIComponent(imgId);
    // Fetch annotation for this image
    try {
        const resp = await fetch('/api/get_label_annotation?filepath=' + encodeURIComponent(imgId));
        const data = await resp.json();
        if (data && Object.prototype.hasOwnProperty.call(data, 'class')) {
            if (data.class) {
                imageSelectedClass[imgId] = data.class;
            } else {
                delete imageSelectedClass[imgId];
            }
        }
    } catch (e) {
        // On error, clear selection
        delete imageSelectedClass[imgId];
    }
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

async function goToPrev() {
    if (isLoading) return;
    if (currentIdx > 0) {
        await saveCurrentImageClassAnnotation();
        currentIdx--;
        updateImage();
        updateNavButtons();
        // Fetch the image list after navigating
        fetchImageList();
    }
}

async function goToNext() {
    if (isLoading) return;
    if (currentIdx < imageIds.length - 1) {
        await saveCurrentImageClassAnnotation();
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

    // --- Add Class Button Logic ---
    const addClassBtn = document.getElementById('add-class-btn');
    const classInput = document.getElementById('class-input');
    if (addClassBtn && classInput) {
        addClassBtn.addEventListener('click', () => {
            const className = classInput.value.trim();
            if (!className) return;
            if (!globalClasses.includes(className)) {
                globalClasses.push(className);
                updateClassListDisplay();
            }
            classInput.value = '';
        });
        classInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                addClassBtn.click();
            }
        });
    }

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
        newScale = Math.max(minScale, Math.min(maxScale, newScale));
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
