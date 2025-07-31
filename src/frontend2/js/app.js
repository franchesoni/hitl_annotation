import { ImageViewer } from './imageViewer.js';
import { ClassManager } from './classManager.js';

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

	// Create the viewer instance
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

	// Create the class manager instance
	const classManager = new ClassManager(classPanel);

	// Fetch the first image from /next endpoint
	let firstImageId = null;
	try {
		const res = await fetch('/next');
		if (!res.ok) {
			console.error('No images available from API.');
			return;
		}
		// The response is an image, get the image id from headers
		firstImageId = res.headers.get('X-Image-Id');
		const blob = await res.blob();
		const imageUrl = URL.createObjectURL(blob);
		viewer.loadImage(imageUrl, firstImageId);
		classManager.setCurrentImageId(firstImageId);
	} catch (e) {
		console.error('Failed to fetch first image:', e);
		return;
	}
});
