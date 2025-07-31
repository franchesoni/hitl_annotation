import { ImageViewer } from './imageViewer.js';

document.addEventListener('DOMContentLoaded', async () => {
	// Get left panel container
	const leftPanel = document.querySelector('.left-panel');
	if (!leftPanel) {
		console.error('Left panel container not found.');
		return;
	}

	// Create the viewer instance, canvas will be created inside leftPanel
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

	// Set canvas size to fit left panel
	const rect = leftPanel.getBoundingClientRect();
	viewer.canvas.width = rect.width * 0.9;
	viewer.canvas.height = rect.height * 0.9;

	// Fetch image IDs from API
	let imageIds = [];
	try {
		const res = await fetch('/api/ids');
		imageIds = await res.json();
	} catch (e) {
		console.error('Failed to fetch image IDs:', e);
		return;
	}

	if (!imageIds.length) {
		console.error('No images available from API.');
		return;
	}

	// Load the first image using /api/sample?id=...
	const firstImageId = imageIds[0];
	const imageUrl = `/api/sample?id=${encodeURIComponent(firstImageId)}`;
	viewer.loadImage(imageUrl, firstImageId);
});
