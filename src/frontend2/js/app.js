import { ImageViewer } from './imageViewer.js';
import { ClassManager } from './classManager.js';
import { API } from './api.js';

document.addEventListener('DOMContentLoaded', async () => {
	// Get left and right panel containers
	const leftPanel = document.querySelector('.left-panel');
	const classPanel = document.querySelector('.right-panel');
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

	// Create the viewer instance
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

	// Helper to load next image from API and update viewer/classManager
	async function loadNextImage() {
		try {
			// Pass the current image filename to API.loadNextImage
			const currentId = classManager.currentImageFilename;
			const { imageUrl, filename } = await api.loadNextImage(currentId);
			viewer.loadImage(imageUrl, filename);
			classManager.setCurrentImageFilename(filename);
		} catch (e) {
			console.error('Failed to fetch next image:', e);
		}
	}

	// Create the class manager instance, passing loadNextImage and api
	const classManager = new ClassManager(classPanel, loadNextImage, api);

	// Fetch the first image from API
	try {
		const { imageUrl, filename } = await api.loadNextImage();
		viewer.loadImage(imageUrl, filename);
		classManager.setCurrentImageFilename(filename);
	} catch (e) {
		console.error('Failed to fetch first image:', e);
		return;
	}
});
