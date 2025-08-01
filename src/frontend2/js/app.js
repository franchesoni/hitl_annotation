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
        const strategySelect = document.getElementById('strategy-select');
        let currentStrategy = strategySelect ? strategySelect.value : null;
        if (strategySelect) {
                strategySelect.addEventListener('change', () => {
                        currentStrategy = strategySelect.value;
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

	// Create the viewer instance
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

	// Helper to load next image from API and update viewer/classManager
        async function loadNextImage() {
                try {
                        const currentId = classManager.currentImageFilename;
                        const { imageUrl, filename, labelClass, labelSource } = await api.loadNextImage(currentId, currentStrategy);
                        viewer.loadImage(imageUrl, filename);
                        const annClass = labelSource === 'annotation' ? labelClass : null;
                        await classManager.setCurrentImageFilename(filename, annClass);
                        await updateStats();
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
                const { imageUrl, filename, labelClass, labelSource } = await api.loadNextImage(null, currentStrategy);
                viewer.loadImage(imageUrl, filename);
                const annClass = labelSource === 'annotation' ? labelClass : null;
                await classManager.setCurrentImageFilename(filename, annClass);
                await updateStats();
        } catch (e) {
                console.error('Failed to fetch first image:', e);
                return;
        }
});
