let annotationRequestId = 0;
async function saveCurrentImageClassAnnotation() {
    if (!currentImageId) return;
    const selectedClass = imageSelectedClass[currentImageId];
    const requestId = ++annotationRequestId;
    if (selectedClass) {
        await fetch('/api/save_label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: currentImageId, class: selectedClass })
        });
    } else {
        await fetch('/api/save_label', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: currentImageId })
        });
    }
    if (requestId === annotationRequestId) {
        // Track annotation history for undo on both save and delete
        annotationHistory.push(currentImageId);
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
const runAIBtn = document.getElementById('run-ai-btn');
const stopAIBtn = document.getElementById('stop-ai-btn');
const aiStatusDiv = document.getElementById('ai-status');
let aiRunning = false;
function updateAIButtons() {
    if (runAIBtn) runAIBtn.style.display = aiRunning ? 'none' : 'inline-block';
    if (stopAIBtn) stopAIBtn.style.display = aiRunning ? 'inline-block' : 'none';
}
updateAIButtons();
const aiArchInput = document.getElementById('ai-arch');
const aiSleepInput = document.getElementById('ai-sleep');
const aiBudgetInput = document.getElementById('ai-budget');
const aiResizeInput = document.getElementById('ai-resize');

let currentImageId = null; // Current image being displayed
let isLoading = false;
let nextImageRequestId = 0;
let accuracyRequestId = 0;
// Zoom & Pan State
let scale = 1;
let panX = 0, panY = 0;
let minScale = 1;
const maxSideScale = 0.75; // Fraction of window size used for canvas and fitting
let dragging = false, startX = 0, startY = 0;
window.panZoomEnabled = true;
window.keyboardShortcutsEnabled = true;
window.autoAdvanceEnabled = true;

// Global class list state
let globalClasses = []; // [class1, class2, ...]
// Per-image selected class
let imageSelectedClass = {}; // { imageId: className }

// Undo history stack
let annotationHistory = [];

if (runAIBtn) {
    runAIBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/run_ai', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    architecture: aiArchInput?.value || 'resnet18',
                    sleep: Number(aiSleepInput?.value || 0),
                    budget: Number(aiBudgetInput?.value || 1000),
                    resize: Number(aiResizeInput?.value || 64)
                })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.status || 'Failed to start AI');
            aiRunning = true;
            if (aiStatusDiv) aiStatusDiv.textContent = data.status;
        } catch (e) {
            if (aiStatusDiv) aiStatusDiv.textContent = e.message;
            aiRunning = e.message.includes('already') ? true : false;
            console.error('Failed to start AI', e);
        } finally {
            updateAIButtons();
        }
    });
}

if (stopAIBtn) {
    stopAIBtn.addEventListener('click', async () => {
        try {
            const res = await fetch('/stop_ai', { method: 'POST' });
            const data = await res.json();
            if (!res.ok) throw new Error(data.status || 'Failed to stop AI');
            aiRunning = false;
            if (aiStatusDiv) aiStatusDiv.textContent = data.status;
        } catch (e) {
            if (aiStatusDiv) aiStatusDiv.textContent = e.message;
            aiRunning = e.message.includes('not running') ? false : true;
            console.error('Failed to stop AI', e);
        } finally {
            updateAIButtons();
        }
    });
}


// =====================
// UI Helpers
// =====================
function setLoading(loading) {
    isLoading = loading;
    updateNavButtons();
    if (overlay) overlay.style.display = loading ? 'flex' : 'none';
    const buttons = document.querySelectorAll('.class-btn');
    buttons.forEach(btn => {
        btn.disabled = loading;
    });
}

function updateNavButtons() {
    // Disable prev button (no navigation backwards in active learning mode)
    if (prevBtn) prevBtn.disabled = true;
    // Next button triggers loading next unlabeled image
    if (nextBtn) nextBtn.disabled = isLoading;
}

function updateIdDisplay() {
    if (idDiv) {
        if (currentImageId) {
            idDiv.innerHTML = `<b>Current Image</b><br><div style="word-break: break-all; font-size: 12px; color: #6c757d;">${currentImageId}</div>`;
        } else {
            idDiv.innerHTML = `<b>No image loaded</b>`;
        }
    }
    updateClassListDisplay();
}

function updatePredictionDisplay(predictionInfo) {
    const predDiv = document.getElementById('prediction-container');
    if (!predDiv) return;
    if (predictionInfo && predictionInfo.class) {
        predDiv.innerHTML = `<span class="prediction-badge">${predictionInfo.class}</span>` +
            (typeof predictionInfo.probability === 'number' ? ` <span style="color:#6c757d;">(${predictionInfo.probability.toFixed(2)})</span>` : '');
    } else {
        predDiv.innerHTML = '<span style="color: #6c757d; font-style: italic;">No prediction available</span>';
    }
}

// Add accuracy display below prediction
async function updateAccuracyDisplay() {
    const accDiv = document.getElementById('accuracy-container');
    if (!accDiv) return;
    const requestId = ++accuracyRequestId;
    try {
        const r = await fetch('/api/accuracy_stats');
        if (requestId !== accuracyRequestId) return;
        const stats = await r.json();
        if (requestId !== accuracyRequestId) return;
        if (typeof stats.accuracy === 'number') {
            accDiv.innerHTML = `<span class="accuracy-badge">${(stats.accuracy * 100).toFixed(1)}%</span> <span style="color:#6c757d;">(${stats.correct}/${stats.tries} correct)</span>`;
        } else {
            accDiv.innerHTML = `<span style="color: #6c757d; font-style: italic;">Not enough data</span>`;
        }
    } catch {
        if (requestId === accuracyRequestId) {
            accDiv.innerHTML = `<span style="color: #dc3545; font-style: italic;">Error loading stats</span>`;
        }
    }
}

function updateClassListDisplay() {
    const classListDiv = document.getElementById('class-list-container');
    if (!classListDiv) return;
    if (globalClasses.length === 0) {
        classListDiv.innerHTML = '<div style="color: #6c757d; font-style: italic;">No classes added yet</div>';
        return;
    }
    const selected = currentImageId ? imageSelectedClass[currentImageId] : undefined;
    
    classListDiv.innerHTML = globalClasses.map((c, index) => {
        const isSelected = c === selected;
        const shortcut = index < 9 ? (index + 1).toString() : (index === 9 ? '0' : '');
        const displayText = shortcut ? `${c} (${shortcut})` : c;
        const disabledAttr = isLoading ? 'disabled' : '';
        return `<button class="class-btn ${isSelected ? 'selected' : ''}" data-class="${c}" ${disabledAttr}>${displayText}</button>`;
    }).join('');

    // Add click listeners for class selection
    Array.from(classListDiv.querySelectorAll('.class-btn')).forEach(btn => {
        btn.addEventListener('click', async () => {
            if (!currentImageId || isLoading) return;
            const className = btn.dataset.class;
            if (imageSelectedClass[currentImageId] === className) {
                delete imageSelectedClass[currentImageId];
            } else {
                imageSelectedClass[currentImageId] = className;
            }
            updateClassListDisplay();
            setLoading(true);
            try {
                await saveCurrentImageClassAnnotation();
                if (window.autoAdvanceEnabled) {
                    await loadNextImage(currentImageId);
                    await updateAccuracyDisplay();
                }
            } catch (err) {
                console.error('Annotation error:', err);
            } finally {
                if (!window.autoAdvanceEnabled) {
                    setLoading(false);
                }
            }
        });
    });
}

// =====================
// Image Logic
// =====================
function drawImageToCanvas() {
    // Get left panel dimensions
    const leftPanel = document.querySelector('.left-panel');
    const leftPanelRect = leftPanel.getBoundingClientRect();
    const maxWidth = leftPanelRect.width * 0.9; // Leave some margin
    const maxHeight = leftPanelRect.height * 0.9; // Leave some margin
    
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
    const leftPanel = document.querySelector('.left-panel');
    const leftPanelRect = leftPanel.getBoundingClientRect();
    const maxWidth = leftPanelRect.width * 0.9;
    const maxHeight = leftPanelRect.height * 0.9;
    
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
    
    // Extract label/prediction data from image response headers if available
    if (img.currentResponse) {
        const imageId = img.currentResponse.headers.get('X-Image-Id');
        const labelClass = img.currentResponse.headers.get('X-Label-Class');
        const labelSource = img.currentResponse.headers.get('X-Label-Source');
        const labelProbability = img.currentResponse.headers.get('X-Label-Probability');
        
        let predictionInfo = null;
        
        // Update current image ID from response
        currentImageId = imageId;
        
        if (labelClass && labelSource === 'annotation') {
            imageSelectedClass[currentImageId] = labelClass;
        } else if (labelClass && labelSource === 'prediction') {
            delete imageSelectedClass[currentImageId];
            predictionInfo = {
                class: labelClass,
                probability: labelProbability ? parseFloat(labelProbability) : undefined
            };
        } else {
            delete imageSelectedClass[currentImageId];
        }
        
        updateIdDisplay();
        updatePredictionDisplay(predictionInfo);
    }
};

async function loadNextImage(currentId) {
    const requestId = ++nextImageRequestId;
    setLoading(true);
    try {
        let url = '/api/next';
        if (currentId) {
            url += '?current_id=' + encodeURIComponent(currentId);
        }
        const response = await fetch(url);
        if (requestId !== nextImageRequestId) return;
        if (!response.ok) {
            if (response.status === 404) {
                alert('No more unlabeled images available!');
            } else {
                alert('Error loading next image.');
            }
            if (requestId === nextImageRequestId) setLoading(false);
            return;
        }
        const blob = await response.blob();
        if (requestId !== nextImageRequestId) return;
        const imageUrl = URL.createObjectURL(blob);
        img.currentResponse = response;
        img.src = imageUrl;
        if (img.previousObjectUrl) URL.revokeObjectURL(img.previousObjectUrl);
        img.previousObjectUrl = imageUrl;
    } catch (e) {
        console.error('Error loading next image:', e);
        alert('Error loading next image. Please try again.');
        if (requestId === nextImageRequestId) setLoading(false);
    }
}

// =====================
// Navigation
// =====================
async function goToPrev() {
    // Disabled in active learning mode - no going back
    return;
}

async function goToNext() {
    if (isLoading) return;
    setLoading(true);
    try {
        await saveCurrentImageClassAnnotation();
        await loadNextImage(currentImageId);
        await updateAccuracyDisplay();
    } catch (err) {
        console.error('Failed to go to next image:', err);
        setLoading(false);
    }
    updateNavButtons();
}

// =====================
// Initialization
// =====================
document.addEventListener('DOMContentLoaded', async () => {
    await loadNextImage(); // Load first unlabeled image
    await updateAccuracyDisplay();
    if (prevBtn) prevBtn.addEventListener('click', goToPrev);
    if (nextBtn) nextBtn.addEventListener('click', goToNext);
    updateNavButtons(); // Set initial button states

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
        // Check if keyboard shortcuts are enabled
        if (!window.keyboardShortcutsEnabled) return;
        
        // Allow number keys 1-9 and 0 for class selection (0 = 10th class)
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        
        let idx = -1;
        if (e.key >= '1' && e.key <= '9') {
            idx = parseInt(e.key, 10) - 1;
        } else if (e.key === '0') {
            idx = 9;
        }
        
        if (idx >= 0 && idx < globalClasses.length) {
            if (!currentImageId || isLoading) return;
            const className = globalClasses[idx];
            if (imageSelectedClass[currentImageId] === className) {
                delete imageSelectedClass[currentImageId];
            } else {
                imageSelectedClass[currentImageId] = className;
            }
            updateClassListDisplay();
            setLoading(true);
            try {
                await saveCurrentImageClassAnnotation();
                if (window.autoAdvanceEnabled) {
                    await loadNextImage(currentImageId);
                    await updateAccuracyDisplay();
                }
            } catch (err) {
                console.error('Annotation error:', err);
            } finally {
                if (!window.autoAdvanceEnabled) {
                    setLoading(false);
                }
            }
            return;
        }
        
        // Navigation shortcuts
        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            goToPrev();
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            goToNext();
        } else if (e.key.toLowerCase() === 'r') {
            e.preventDefault();
            if (typeof resetViewToImage === 'function' && typeof drawImageToCanvas === 'function') {
                resetViewToImage();
                drawImageToCanvas();
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

    // Undo button logic
    const undoBtn = document.getElementById('undo-btn');
    if (undoBtn) {
        undoBtn.addEventListener('click', async () => {
            if (annotationHistory.length === 0) {
                alert('No annotation to undo.');
                return;
            }
            const lastAnnotatedId = annotationHistory.pop();
            if (!lastAnnotatedId) return;
            setLoading(true);
            try {
                // Fetch the sample for the previous image
                const response = await fetch(`/api/sample?id=${encodeURIComponent(lastAnnotatedId)}`);
                if (!response.ok) {
                    alert('Could not load previous annotation.');
                    setLoading(false);
                    return;
                }
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                img.currentResponse = response;
                img.src = imageUrl;
                if (img.previousObjectUrl) URL.revokeObjectURL(img.previousObjectUrl);
                img.previousObjectUrl = imageUrl;
            } catch (e) {
                console.error('Error loading previous annotation:', e);
                alert('Error loading previous annotation.');
                setLoading(false);
            }
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
        // Configurable UI Elements - Set defaults to only show Image ID and Class Input
        'check-zoom-pan-reset': false, // Disabled by default
        'check-sequential-nav': false, // Disabled by default
        'check-image-id': true, // Enabled by default
        'check-class-input': true, // Enabled by default
        'check-prediction': false, // Disabled by default
        
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
    
    // Set initial section visibility based on checkbox states
    const imageInfoSection = document.querySelector('.control-section:has(#image-ids)');
    const navigationSection = document.getElementById('navigation-section');
    const classSection = document.querySelector('.control-section:has(#class-input)');
    const predictionSection = document.querySelector('.control-section:has(#prediction-container)');
    const performanceSection = document.querySelector('.control-section:has(#accuracy-container)');
    
    if (imageInfoSection) {
        imageInfoSection.style.display = checks['check-image-id'] ? 'block' : 'none';
    }
    if (navigationSection) {
        const showNav = checks['check-sequential-nav'] || checks['check-zoom-pan-reset'];
        navigationSection.style.display = showNav ? 'block' : 'none';
        
        const navButtons = navigationSection.querySelector('.nav-buttons');
        if (navButtons) {
            navButtons.style.display = checks['check-sequential-nav'] ? 'flex' : 'none';
        }
    }
    if (classSection) {
        classSection.style.display = checks['check-class-input'] ? 'block' : 'none';
    }
    if (predictionSection) {
        predictionSection.style.display = checks['check-prediction'] ? 'block' : 'none';
    }
    if (performanceSection) {
        performanceSection.style.display = checks['check-performance-curve'] ? 'block' : 'none';
    }
    
    // Set initial reset button visibility based on pan/zoom state
    const resetBtn = document.getElementById('reset-view-btn');
    if (resetBtn) {
        resetBtn.style.display = checks['check-zoom-pan-reset'] ? 'block' : 'none';
    }
    
    // Remove undoBtn display logic
    // Only control undoSection visibility
    const undoSection = document.getElementById('undo-section');
    if (undoSection) {
        undoSection.style.display = checks['check-undo'] ? 'block' : 'none';
    }
    
    // Set panZoomEnabled based on checkbox state
    window.panZoomEnabled = checks['check-zoom-pan-reset'];
    window.keyboardShortcutsEnabled = checks['check-keyboard'];
    window.autoAdvanceEnabled = checks['check-auto-advance'];
}

// Add interactive functionality for specific checklist items
function setupChecklistInteractions() {
    // Image ID Display toggle
    const imageIdCheckbox = document.getElementById('check-image-id');
    if (imageIdCheckbox) {
        imageIdCheckbox.addEventListener('change', () => {
            const imageInfoSection = document.querySelector('.control-section:has(#image-ids)');
            if (imageInfoSection) {
                imageInfoSection.style.display = imageIdCheckbox.checked ? 'block' : 'none';
                
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
            if (typeof window.panZoomEnabled !== 'undefined') {
                window.panZoomEnabled = zoomPanCheckbox.checked;
                
                const resetBtn = document.getElementById('reset-view-btn');
                if (resetBtn) {
                    resetBtn.style.display = window.panZoomEnabled ? 'block' : 'none';
                }

                if (!window.panZoomEnabled) {
                    if (typeof resetViewToImage === 'function' && typeof drawImageToCanvas === 'function') {
                        resetViewToImage();
                        drawImageToCanvas();
                    }
                }
                
                const label = zoomPanCheckbox.nextElementSibling;
                if (label) {
                    label.className = zoomPanCheckbox.checked ? 'status-implemented' : 'status-partial';
                }
            }
            
            // Update navigation section visibility
            const navigationSection = document.getElementById('navigation-section');
            const sequentialNavCheckbox = document.getElementById('check-sequential-nav');
            if (navigationSection && sequentialNavCheckbox) {
                const showNav = zoomPanCheckbox.checked || sequentialNavCheckbox.checked;
                navigationSection.style.display = showNav ? 'block' : 'none';
            }
        });
    }
    
    // Sequential Navigation toggle
    const sequentialNavCheckbox = document.getElementById('check-sequential-nav');
    if (sequentialNavCheckbox) {
        sequentialNavCheckbox.addEventListener('change', () => {
            const navButtons = document.querySelector('.nav-buttons');
            if (navButtons) {
                navButtons.style.display = sequentialNavCheckbox.checked ? 'flex' : 'none';
            }
            
            const label = sequentialNavCheckbox.nextElementSibling;
            if (label) {
                label.className = sequentialNavCheckbox.checked ? 'status-implemented' : 'status-partial';
            }
            
            // Update navigation section visibility
            const navigationSection = document.getElementById('navigation-section');
            const zoomPanCheckbox = document.getElementById('check-zoom-pan-reset');
            if (navigationSection && zoomPanCheckbox) {
                const showNav = sequentialNavCheckbox.checked || zoomPanCheckbox.checked;
                navigationSection.style.display = showNav ? 'block' : 'none';
            }
        });
    }
    
    // Class input toggle (controls entire class management section)
    const classInputCheckbox = document.getElementById('check-class-input');
    if (classInputCheckbox) {
        classInputCheckbox.addEventListener('change', () => {
            const classSection = document.querySelector('.control-section:has(#class-input)');
            if (classSection) {
                classSection.style.display = classInputCheckbox.checked ? 'block' : 'none';
                
                // Update the status color based on new state
                const label = classInputCheckbox.nextElementSibling;
                if (label) {
                    label.className = classInputCheckbox.checked ? 'status-implemented' : 'status-partial';
                }
            }
        });
    }
    
    // Prediction display toggle (controls entire prediction section)
    const predictionCheckbox = document.getElementById('check-prediction');
    if (predictionCheckbox) {
        predictionCheckbox.addEventListener('change', () => {
            const predictionSection = document.querySelector('.control-section:has(#prediction-container)');
            if (predictionSection) {
                predictionSection.style.display = predictionCheckbox.checked ? 'block' : 'none';
                
                // Update the status color based on new state
                const label = predictionCheckbox.nextElementSibling;
                if (label) {
                    label.className = predictionCheckbox.checked ? 'status-implemented' : 'status-partial';
                }
            }
        });
    }
    
    // Performance curve toggle (controls entire model performance section)
    const performanceCurveCheckbox = document.getElementById('check-performance-curve');
    if (performanceCurveCheckbox) {
        performanceCurveCheckbox.addEventListener('change', () => {
            const performanceSection = document.querySelector('.control-section:has(#accuracy-container)');
            if (performanceSection) {
                performanceSection.style.display = performanceCurveCheckbox.checked ? 'block' : 'none';
                
                // Update the status color based on new state
                const label = performanceCurveCheckbox.nextElementSibling;
                if (label) {
                    label.className = performanceCurveCheckbox.checked ? 'status-implemented' : 'status-partial';
                }
            }
        });
    }
    
    // Undo button toggle
    const undoCheckbox = document.getElementById('check-undo');
    if (undoCheckbox) {
        undoCheckbox.addEventListener('change', () => {
            const undoSection = document.getElementById('undo-section');
            if (undoSection) {
                undoSection.style.display = undoCheckbox.checked ? 'block' : 'none';
            }
            const label = undoCheckbox.nextElementSibling;
            if (label) {
                label.className = undoCheckbox.checked ? 'status-implemented' : 'status-partial';
            }
        });
    }
    
    // Keyboard shortcuts toggle
    const keyboardCheckbox = document.getElementById('check-keyboard');
    if (keyboardCheckbox) {
        keyboardCheckbox.addEventListener('change', () => {
            window.keyboardShortcutsEnabled = keyboardCheckbox.checked;
            
            // Update the status color based on new state
            const label = keyboardCheckbox.nextElementSibling;
            if (label) {
                label.className = keyboardCheckbox.checked ? 'status-implemented' : 'status-partial';
            }
        });
    }
    
    // Auto-advance toggle
    const autoAdvanceCheckbox = document.getElementById('check-auto-advance');
    if (autoAdvanceCheckbox) {
        autoAdvanceCheckbox.addEventListener('change', () => {
            window.autoAdvanceEnabled = autoAdvanceCheckbox.checked;
            
            // Update the status color based on new state
            const label = autoAdvanceCheckbox.nextElementSibling;
            if (label) {
                label.className = autoAdvanceCheckbox.checked ? 'status-implemented' : 'status-partial';
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
