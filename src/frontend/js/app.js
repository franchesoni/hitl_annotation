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
    updateAccuracyDisplay();
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
window.panZoomEnabled = true;

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

function updatePredictionDisplay(predictionInfo) {
    const predDiv = document.getElementById('prediction-container');
    if (!predDiv) return;
    if (predictionInfo && predictionInfo.class) {
        predDiv.innerHTML = `<b>Prediction:</b> <span style="background:#ffe0b2;padding:2px 8px;border-radius:4px;">${predictionInfo.class}</span>` +
            (typeof predictionInfo.probability === 'number' ? ` <span style="color:#888;">(prob: ${predictionInfo.probability.toFixed(2)})</span>` : '');
    } else {
        predDiv.innerHTML = '';
    }
}

// Add accuracy display below prediction
function updateAccuracyDisplay() {
    const accDiv = document.getElementById('accuracy-container');
    if (!accDiv) return;
    fetch('/api/accuracy_stats')
        .then(r => r.json())
        .then(stats => {
            if (typeof stats.accuracy === 'number') {
                accDiv.innerHTML = `<b>Model Accuracy:</b> <span style="background:#c8e6c9;padding:2px 8px;border-radius:4px;">${(stats.accuracy * 100).toFixed(1)}%</span> <span style="color:#888;">(${stats.correct}/${stats.tries} correct)</span>`;
            } else {
                accDiv.innerHTML = `<b>Model Accuracy:</b> <i>Not enough data</i>`;
            }
        })
        .catch(() => {
            accDiv.innerHTML = `<b>Model Accuracy:</b> <i>Error</i>`;
        });
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
        btn.addEventListener('click', async () => {
            if (!imgId) return;
            const className = btn.dataset.class;
            if (imageSelectedClass[imgId] === className) {
                // Unselect if already selected
                delete imageSelectedClass[imgId];
            } else {
                imageSelectedClass[imgId] = className;
            }
            updateClassListDisplay();
            // Save annotation and go to next image, ensuring order
            await saveCurrentImageClassAnnotation();
            if (currentIdx < imageIds.length - 1) {
                currentIdx++;
                await updateImage();
                updateNavButtons();
                fetchImageList();
            }
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
    // Fetch annotation or prediction for this image
    let predictionInfo = null;
    try {
        const resp = await fetch('/api/get_label_or_prediction?filepath=' + encodeURIComponent(imgId));
        const data = await resp.json();
        if (data && Object.prototype.hasOwnProperty.call(data, 'class')) {
            if (data.class && data.source === 'annotation') {
                imageSelectedClass[imgId] = data.class;
                predictionInfo = null;
            } else if (data.class && data.source === 'prediction') {
                delete imageSelectedClass[imgId];
                predictionInfo = {
                    class: data.class,
                    probability: data.probability
                };
            } else {
                delete imageSelectedClass[imgId];
                predictionInfo = null;
            }
        }
    } catch (e) {
        // On error, clear selection
        delete imageSelectedClass[imgId];
        predictionInfo = null;
    }
    updateIdDisplay();
    updatePredictionDisplay(predictionInfo);
    updateAccuracyDisplay();
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

    // --- Keyboard shortcuts for class selection ---
    document.addEventListener('keydown', async (e) => {
        // Allow number keys 1-9 and 0 for class selection (0 = 10th class)
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        let idx = -1;
        if (e.key >= '1' && e.key <= '9') {
            idx = parseInt(e.key, 10) - 1;
        } else if (e.key === '0') {
            idx = 9;
        }
        if (idx >= 0 && idx < globalClasses.length) {
            const imgId = imageIds[currentIdx];
            if (!imgId) return;
            const className = globalClasses[idx];
            if (imageSelectedClass[imgId] === className) {
                delete imageSelectedClass[imgId];
            } else {
                imageSelectedClass[imgId] = className;
            }
            updateClassListDisplay();
            await saveCurrentImageClassAnnotation();
            if (currentIdx < imageIds.length - 1) {
                currentIdx++;
                await updateImage();
                updateNavButtons();
                fetchImageList();
            }
        }
    });
    // --- Zoom and Pan ---
    // Mouse wheel zoom (zoom around cursor)
    canvas.addEventListener('wheel', (e) => {
        if (!window.panZoomEnabled) return;
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
        if (!window.panZoomEnabled) return;
        dragging = true;
        startX = e.clientX;
        startY = e.clientY;
        canvas.setPointerCapture(e.pointerId);
        e.preventDefault();
    });
    canvas.addEventListener('pointermove', (e) => {
        if (!window.panZoomEnabled) return;
        if (!dragging) return;
        panX += e.clientX - startX;
        panY += e.clientY - startY;
        startX = e.clientX;
        startY = e.clientY;
        drawImageToCanvas();
        e.preventDefault();
    });
    canvas.addEventListener('pointerup', (e) => {
        if (!window.panZoomEnabled) return;
        dragging = false;
        canvas.releasePointerCapture(e.pointerId);
    });
    window.addEventListener('pointerup', () => { dragging = false; });

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

// =====================
// Developer Panel Functionality
// =====================

// Toggle developer checklist
function setupDeveloperPanel() {
    const devToggle = document.getElementById('dev-toggle');
    const devChecklist = document.getElementById('dev-checklist');
    
    if (devToggle && devChecklist) {
        devToggle.addEventListener('click', () => {
            devChecklist.classList.toggle('collapsed');
        });
    }
}

// Update checklist based on actual implementation
function updateChecklistStatus() {
    // Check what's actually implemented in the DOM/JS
    const checks = {
        // Configurable UI Elements
        'check-zoom-pan-reset': (typeof window.panZoomEnabled !== 'undefined') ? window.panZoomEnabled : true,
        'check-sequential-nav': !!document.getElementById('prev-btn') && document.getElementById('prev-btn').style.display !== 'none',
        'check-image-id': !!document.getElementById('image-ids') && document.getElementById('image-ids').style.display !== 'none',
        'check-class-input': !!document.getElementById('class-input'),
        'check-class-list': !!document.getElementById('class-list-container'),
        'check-prediction': !!document.getElementById('prediction-container'),
        
        // Features implemented in JS
        'check-keyboard': true, // Keyboard shortcuts in app.js
        'check-auto-advance': true, // Auto-advance in class selection
        
        // Missing features
        'check-delete-class': false,
        'check-undo': false,
        'check-training-curve': false,
        'check-performance-curve': false,
        'check-architecture': false,
        'check-bbox-drawing': false,
        'check-bbox-editing': false,
        'check-bbox-class': false,
        'check-bbox-removal': false,
        'check-bbox-list': false,
        'check-skip': false,
        'check-save': false,
        'check-export': false,
        'check-config': false
    };
    
    // Update checkboxes and labels
    Object.entries(checks).forEach(([id, isImplemented]) => {
        const checkbox = document.getElementById(id);
        const label = checkbox?.nextElementSibling;
        if (checkbox && label) {
            checkbox.checked = isImplemented;
            
            // Update status colors based on actual implementation
            label.className = '';
            if (isImplemented) {
                label.className = 'status-implemented';
            } else if (label.textContent.includes('◇') || label.textContent.includes('🎛️')) {
                // Configurable features
                label.className = 'status-partial';
            } else {
                label.className = 'status-missing';
            }
        }
    });
    
    // Set initial reset button visibility based on pan/zoom state
    const resetBtn = document.getElementById('reset-view-btn');
    if (resetBtn && typeof window.panZoomEnabled !== 'undefined') {
        resetBtn.style.display = window.panZoomEnabled ? 'inline-block' : 'none';
    }
}

// Add interactive functionality for specific checklist items
function setupChecklistInteractions() {
    // Image ID Display toggle
    const imageIdCheckbox = document.getElementById('check-image-id');
    if (imageIdCheckbox) {
        imageIdCheckbox.addEventListener('change', () => {
            const imageIdDiv = document.getElementById('image-ids');
            if (imageIdDiv) {
                imageIdDiv.style.display = imageIdCheckbox.checked ? 'block' : 'none';
                
                // Update the status color based on new state
                const label = imageIdCheckbox.nextElementSibling;
                if (label) {
                    label.className = imageIdCheckbox.checked ? 'status-implemented' : 'status-partial';
                }
            }
        });
    }
    
    // Zoom/Pan/Reset View toggle (replaces the old toggle button functionality)
    const zoomPanCheckbox = document.getElementById('check-zoom-pan-reset');
    if (zoomPanCheckbox) {
        zoomPanCheckbox.addEventListener('change', () => {
            // Access the global panZoomEnabled variable from app.js
            if (typeof window.panZoomEnabled !== 'undefined') {
                window.panZoomEnabled = zoomPanCheckbox.checked;
                
                // Show/hide reset button based on zoom/pan state
                const resetBtn = document.getElementById('reset-view-btn');
                if (resetBtn) {
                    resetBtn.style.display = window.panZoomEnabled ? 'inline-block' : 'none';
                }

                // If zoom is disabled, reset the view
                if (!window.panZoomEnabled) {
                    if (typeof resetViewToImage === 'function' && typeof drawImageToCanvas === 'function') {
                        resetViewToImage();
                        drawImageToCanvas();
                    }
                }
                
                // Update the status color based on new state
                const label = zoomPanCheckbox.nextElementSibling;
                if (label) {
                    label.className = zoomPanCheckbox.checked ? 'status-implemented' : 'status-partial';
                }
            }
        });
    }
    
    // Sequential Navigation toggle (controls prev/next buttons)
    const sequentialNavCheckbox = document.getElementById('check-sequential-nav');
    if (sequentialNavCheckbox) {
        sequentialNavCheckbox.addEventListener('change', () => {
            const prevBtn = document.getElementById('prev-btn');
            const nextBtn = document.getElementById('next-btn');
            
            // Show/hide both navigation buttons
            if (prevBtn) {
                prevBtn.style.display = sequentialNavCheckbox.checked ? 'inline-block' : 'none';
            }
            if (nextBtn) {
                nextBtn.style.display = sequentialNavCheckbox.checked ? 'inline-block' : 'none';
            }
            
            // Update the status color based on new state
            const label = sequentialNavCheckbox.nextElementSibling;
            if (label) {
                label.className = sequentialNavCheckbox.checked ? 'status-implemented' : 'status-partial';
            }
        });
    }
}

// Initialize developer panel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        setupDeveloperPanel();
        updateChecklistStatus();
        setupChecklistInteractions();
    }, 100); // Wait for all elements to be ready
});
