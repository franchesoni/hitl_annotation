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
                        if (stats && stats.image) {
                                statsDiv.textContent = `${stats.image} (${stats.annotated}/${stats.total})`;
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
                        const { imageUrl, filename, labelClass, labelSource } = await api.loadNextImage(currentId);
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
                const { imageUrl, filename, labelClass, labelSource } = await api.loadNextImage();
                viewer.loadImage(imageUrl, filename);
                const annClass = labelSource === 'annotation' ? labelClass : null;
                await classManager.setCurrentImageFilename(filename, annClass);
                await updateStats();
        } catch (e) {
                console.error('Failed to fetch first image:', e);
                return;
        }
});
