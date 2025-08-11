import { ImageViewer } from './imageViewer.js';
import { ClassManager } from './classManager.js';
import { API } from './api.js';
import { UndoManager } from './undoManager.js';

document.addEventListener('DOMContentLoaded', async () => {
        // Get left and right panel containers
        const leftPanel = document.querySelector('.left-panel');
        const classPanel = document.querySelector('#class-manager');
        const undoBtn = document.getElementById('undo-btn');
        const exportDBBtn = document.getElementById('export-db-btn');
        const runBtn = document.getElementById('run-ai-btn');
        const stopBtn = document.getElementById('stop-ai-btn');
        const aiStatus = document.getElementById('ai-status');
        let aiRunning = false;
        function updateAIButtons() {
                if (runBtn) runBtn.style.display = aiRunning ? 'none' : 'inline-block';
                if (stopBtn) stopBtn.style.display = aiRunning ? 'inline-block' : 'none';
        }
        const archInput = document.getElementById('ai-arch');
        const sleepInput = document.getElementById('ai-sleep');
        const budgetInput = document.getElementById('ai-budget');
        const resizeInput = document.getElementById('ai-resize');
        const statsDiv = document.getElementById('stats-display');
        const predictionDiv = document.getElementById('prediction-display');
        const trainingCanvas = document.getElementById('training-curve');
        const strategySelect = document.getElementById('strategy-select');
        const accuracySlider = document.getElementById('accuracy-slider');
        const accuracyValue = document.getElementById('accuracy-slider-value');
        let accuracyPct = accuracySlider ? Number(accuracySlider.value) : 100;
        if (accuracyValue) accuracyValue.textContent = `${accuracyPct}%`;
        if (accuracySlider) {
                accuracySlider.addEventListener('input', () => {
                        accuracyPct = Number(accuracySlider.value);
                        if (accuracyValue) accuracyValue.textContent = `${accuracyPct}%`;
                        updateStats();
                });
        }
        const specificClassSelect = document.getElementById('specific-class-select');
        const specificClassLabel = document.getElementById('specific-class-label');
        let currentStrategy = strategySelect ? strategySelect.value : null;
        let currentSpecificClass = specificClassSelect ? specificClassSelect.value : null;
        function toggleSpecificClassSelect() {
                const show = currentStrategy === 'specific_class';
                if (specificClassSelect) specificClassSelect.style.display = show ? 'inline-block' : 'none';
                if (specificClassLabel) specificClassLabel.style.display = show ? 'inline-block' : 'none';
        }
        toggleSpecificClassSelect();
        if (strategySelect) {
                strategySelect.addEventListener('change', () => {
                        currentStrategy = strategySelect.value;
                        toggleSpecificClassSelect();
                });
        }
        if (specificClassSelect) {
                specificClassSelect.addEventListener('change', () => {
                        currentSpecificClass = specificClassSelect.value;
                });
        }
        if (!leftPanel) {
                console.error('Left panel container not found.');
                return;
        }
        if (!classPanel) {
                console.error('Class list container not found.');
                return;
        }

        // Create the API instance
        const api = new API();
        updateAIButtons();
        if (exportDBBtn) {
                exportDBBtn.addEventListener('click', () => api.exportDB());
        }
        if (runBtn) {
                runBtn.addEventListener('click', async () => {
                        try {
                                const res = await api.runAI({
                                        architecture: archInput?.value || 'resnet18',
                                        sleep: Number(sleepInput?.value || 0),
                                        budget: Number(budgetInput?.value || 1000),
                                        resize: Number(resizeInput?.value || 64)
                                });
                                aiRunning = true;
                                if (aiStatus) aiStatus.textContent = res.status;
                        } catch (e) {
                                if (aiStatus) aiStatus.textContent = e.message;
                                aiRunning = e.message.includes('already') ? true : false;
                                console.error('Failed to start AI:', e);
                        } finally {
                                updateAIButtons();
                        }
                });
        }
        if (stopBtn) {
                stopBtn.addEventListener('click', async () => {
                        try {
                                const res = await api.stopAI();
                                aiRunning = false;
                                if (aiStatus) aiStatus.textContent = res.status;
                        } catch (e) {
                                if (aiStatus) aiStatus.textContent = e.message;
                                aiRunning = e.message.includes('not running') ? false : true;
                                console.error('Failed to stop AI:', e);
                        } finally {
                                updateAIButtons();
                        }
                });
        }

        document.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'e') {
                        e.preventDefault();
                        api.exportDB();
                }
        });

        function updatePrediction(labelClass, labelProbability, labelSource) {
                if (!predictionDiv) return;
                if (labelSource === 'prediction' && labelClass) {
                        const prob = labelProbability ? Number(labelProbability) : null;
                        const pct = prob !== null && !isNaN(prob) ? (prob * 100).toFixed(1) : null;
                        const text = pct !== null ? `${labelClass} (${pct}%)` : labelClass;
                        predictionDiv.innerHTML = `<b>Prediction:</b> <span class="prediction-badge">${text}</span>`;
                        if (classManager && typeof classManager.setPrediction === 'function') {
                                classManager.setPrediction(labelClass);
                        }
                } else {
                        predictionDiv.innerHTML = '';
                        if (classManager && typeof classManager.setPrediction === 'function') {
                                classManager.setPrediction(null);
                        }
                }
        }

        let statsRequestId = 0;
        async function updateStats() {
                if (!statsDiv) return;
                const requestId = ++statsRequestId;
                try {
                        const stats = await api.getStats(accuracyPct);
                        if (requestId !== statsRequestId) return;
                        if (stats) {
                                let html = '';
                                if (stats.image) {
                                        html += `<div><b>Last image:</b> ${stats.image}</div>`;
                                }
                                html += `<div><b>Annotated:</b> ${stats.annotated}/${stats.total}</div>`;

                                if (stats.annotation_counts) {
                                        const counts = Object.entries(stats.annotation_counts)
                                                .map(([cls, n]) => `<div>${cls}: ${n}</div>`)
                                                .join('');
                                        html += `<div><b>Annotations per class:</b>${counts}</div>`;
                                }

                                html += `<div><b>Tries:</b> ${stats.tries}</div>`;
                                html += `<div><b>Correct:</b> ${stats.correct}</div>`;

                                if (typeof stats.accuracy === 'number') {
                                        const pct = (stats.accuracy * 100).toFixed(1);
                                        html += `<div><b>Val Accuracy:</b> <span class="accuracy-badge">${pct}%</span></div>`;
                                } else {
                                        html += `<div><b>Val Accuracy:</b> <span class="accuracy-badge">0%</span></div>`;
                                }

                                statsDiv.innerHTML = html;
                        }
                } catch (e) {
                        if (requestId === statsRequestId) {
                                console.error('Failed to fetch stats:', e);
                        }
                }
        }

        let trainingRequestId = 0;
        async function updateTrainingCurve() {
                if (!trainingCanvas) return;
                const requestId = ++trainingRequestId;
                try {
                        const data = await api.getTrainingStats();
                        if (requestId !== trainingRequestId) return;
                        drawCurve(trainingCanvas, data.map(d => ({x: d.epoch, y: d.accuracy ?? 0})));
                } catch (e) {
                        if (requestId === trainingRequestId) {
                                console.error('Failed to fetch training stats:', e);
                        }
                }
        }

        function drawCurve(canvas, points) {
                const ctx = canvas.getContext('2d');
                const w = canvas.width;
                const h = canvas.height;
                ctx.clearRect(0, 0, w, h);
                if (!points || points.length === 0) return;
                const padding = 30;
                const maxX = points[points.length - 1].x || 1;
                const maxY = 1; // accuracy is in [0,1]

                // axes
                ctx.strokeStyle = '#ccc';
                ctx.beginPath();
                ctx.moveTo(padding, padding);
                ctx.lineTo(padding, h - padding);
                ctx.lineTo(w - padding, h - padding);
                ctx.stroke();

                ctx.fillStyle = '#e0e0e0';
                ctx.font = '12px Roboto, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('Epoch', w / 2, h - 5);
                ctx.save();
                ctx.translate(padding - 25, h / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.fillText('Val Accuracy', 0, 0);
                ctx.restore();

                // tick marks
                const yTicks = [0, 0.5, 1];
                yTicks.forEach(t => {
                        const y = h - padding - t * (h - 2 * padding);
                        ctx.beginPath();
                        ctx.moveTo(padding - 5, y);
                        ctx.lineTo(padding, y);
                        ctx.stroke();
                        ctx.textAlign = 'right';
                        ctx.fillText(t.toString(), padding - 7, y + 4);
                });

                const step = Math.max(1, Math.floor(maxX / 5));
                for (let t = 0; t <= maxX; t += step) {
                        const x = padding + (t / maxX) * (w - 2 * padding);
                        ctx.beginPath();
                        ctx.moveTo(x, h - padding);
                        ctx.lineTo(x, h - padding + 5);
                        ctx.stroke();
                        ctx.textAlign = 'center';
                        ctx.fillText(t.toString(), x, h - padding + 15);
                }

                // curve
                ctx.strokeStyle = '#007acc';
                ctx.beginPath();
                points.forEach((p, idx) => {
                        const x = padding + (p.x / maxX) * (w - 2 * padding);
                        const y = h - padding - (p.y / maxY) * (h - 2 * padding);
                        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
        }

	// Create the viewer instance
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

        // Helper to load next image from API and update viewer/classManager
        let nextImageRequestId = 0;
        async function loadNextImage() {
                const requestId = ++nextImageRequestId;
                if (classManager && typeof classManager.setLoading === 'function') {
                        classManager.setLoading(true);
                }
                try {
                        const currentId = classManager.currentImageFilename;
                        const { imageUrl, filename, labelClass, labelSource, labelProbability } = await api.loadNextImage(currentId, currentStrategy, currentSpecificClass);
                        if (requestId !== nextImageRequestId) {
                                URL.revokeObjectURL(imageUrl);
                                return;
                        }
                        viewer.loadImage(imageUrl, filename);
                        const annClass = labelSource === 'annotation' ? labelClass : null;
                        await classManager.setCurrentImageFilename(filename, annClass);
                        updatePrediction(labelClass, labelProbability, labelSource);
                        await updateStats();
                        await updateTrainingCurve();
                } catch (e) {
                        console.error('Failed to fetch next image:', e);
                } finally {
                        if (requestId === nextImageRequestId && classManager && typeof classManager.setLoading === 'function') {
                                classManager.setLoading(false);
                        }
                }
        }

	// Create the class manager instance, passing loadNextImage and api
        const classManager = new ClassManager(classPanel, loadNextImage, api);
        const undoManager = new UndoManager(api, viewer, classManager, updatePrediction);
        if (undoBtn) {
                undoBtn.addEventListener('click', () => undoManager.undo());
        }
        classManager.setOnClassChange((filename, cls) => {
                undoManager.record(filename);
        });
        classManager.setOnClassesUpdate((classes) => {
                if (specificClassSelect) {
                        const previous = specificClassSelect.value;
                        specificClassSelect.innerHTML = classes.map(c => `<option value="${c}">${c}</option>`).join('');
                        if (classes.includes(previous)) {
                                specificClassSelect.value = previous;
                        } else if (classes.length > 0) {
                                specificClassSelect.selectedIndex = 0;
                        }
                        currentSpecificClass = specificClassSelect.value || null;
                }
        });

        try {
                await loadNextImage();
        } catch (e) {
                console.error('Failed to fetch first image:', e);
                return;
        }

});
