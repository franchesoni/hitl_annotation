import { ImageViewer } from './imageViewer.js';
import { ClassManager } from './classManager.js';
import { API } from './api.js';
import { UndoManager } from './undoManager.js';

document.addEventListener('DOMContentLoaded', async () => {
        // Get left and right panel containers
        const leftPanel = document.querySelector('.left-panel');
        const classPanel = document.querySelector('#class-manager');
        const undoBtn = document.getElementById('undo-btn');
        const statsDiv = document.getElementById('stats-display');
        const trainingCanvas = document.getElementById('training-curve');
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

        async function updateStats() {
                if (!statsDiv) return;
                try {
                        const stats = await api.getStats();
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
                                        html += `<div><b>Accuracy:</b> <span class="accuracy-badge">${pct}%</span></div>`;
                                } else {
                                        html += `<div><b>Accuracy:</b> <span class="accuracy-badge">0%</span></div>`;
                                }

                                statsDiv.innerHTML = html;
                        }
                } catch (e) {
                        console.error('Failed to fetch stats:', e);
                }
        }

        async function updateTrainingCurve() {
                if (!trainingCanvas) return;
                try {
                        const data = await api.getTrainingStats();
                        drawCurve(trainingCanvas, data.map(d => ({x: d.epoch, y: d.accuracy ?? 0})));
                } catch (e) {
                        console.error('Failed to fetch training stats:', e);
                }
        }

        function drawCurve(canvas, points) {
                const ctx = canvas.getContext('2d');
                const w = canvas.width;
                const h = canvas.height;
                ctx.clearRect(0, 0, w, h);
                if (!points || points.length === 0) return;
                const padding = 20;
                const maxX = points[points.length - 1].x || 1;
                ctx.strokeStyle = '#ccc';
                ctx.beginPath();
                ctx.moveTo(padding, padding);
                ctx.lineTo(padding, h - padding);
                ctx.lineTo(w - padding, h - padding);
                ctx.stroke();

                ctx.strokeStyle = '#007acc';
                ctx.beginPath();
                points.forEach((p, idx) => {
                        const x = padding + (p.x / maxX) * (w - 2 * padding);
                        const y = h - padding - (p.y) * (h - 2 * padding);
                        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                });
                ctx.stroke();
        }

	// Create the viewer instance
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

	// Helper to load next image from API and update viewer/classManager
        async function loadNextImage() {
                try {
                        const currentId = classManager.currentImageFilename;
                        const { imageUrl, filename, labelClass, labelSource } = await api.loadNextImage(currentId);
                        viewer.loadImage(imageUrl, filename);
                        const annClass = labelSource === 'annotation' ? labelClass : null;
                        await classManager.setCurrentImageFilename(filename, annClass);
                        await updateStats();
                        await updateTrainingCurve();
                } catch (e) {
                        console.error('Failed to fetch next image:', e);
                }
        }

	// Create the class manager instance, passing loadNextImage and api
        const classManager = new ClassManager(classPanel, loadNextImage, api);
        const undoManager = new UndoManager(api, viewer, classManager);
        if (undoBtn) {
                undoBtn.addEventListener('click', () => undoManager.undo());
        }
        classManager.setOnClassChange((filename, cls) => {
                undoManager.record(filename);
        });

	// Fetch the first image from API
        try {
                const { imageUrl, filename, labelClass, labelSource } = await api.loadNextImage();
                viewer.loadImage(imageUrl, filename);
                const annClass = labelSource === 'annotation' ? labelClass : null;
                await classManager.setCurrentImageFilename(filename, annClass);
                await updateStats();
                await updateTrainingCurve();
        } catch (e) {
                console.error('Failed to fetch first image:', e);
                return;
        }

        updateTrainingCurve();
        setInterval(updateTrainingCurve, 5000);
});
