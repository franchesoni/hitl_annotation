import { ImageViewer } from './imageViewer.js';
import { ClassManager } from './classManager.js';

document.addEventListener('DOMContentLoaded', async () => {
	// Keyboard shortcuts for class selection
	document.addEventListener('keydown', (e) => {
		// Only trigger if not typing in an input/textarea
		if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
		let idx = -1;
		if (e.key >= '1' && e.key <= '9') {
			idx = parseInt(e.key, 10) - 1;
		} else if (e.key === '0') {
			idx = 9;
		}
		if (idx >= 0 && idx < classManager.globalClasses.length) {
			const className = classManager.globalClasses[idx];
			console.log(`Keyboard shortcut: selected class '${className}'`);
			classManager.addClass(className); // Ensure class exists (redundant, but safe)
			classManager.imageSelectedClass[classManager.currentImageId] = className;
			if (classManager.onClassChange) classManager.onClassChange(classManager.currentImageId, className);
			classManager.render();
		}
	});
	// Get left and right panel containers
	const leftPanel = document.querySelector('.left-panel');
	const rightPanel = document.querySelector('.right-panel');
	if (!leftPanel) {
		console.error('Left panel container not found.');
		return;
	}
	if (!rightPanel) {
		console.error('Class list container not found.');
		return;
	}

	// Create the viewer instance
	const viewer = new ImageViewer(leftPanel, 'loading-overlay', 'c');

	// Create the class manager instance
	const classManager = new ClassManager(rightPanel);

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
	classManager.setCurrentImageId(firstImageId);
});
