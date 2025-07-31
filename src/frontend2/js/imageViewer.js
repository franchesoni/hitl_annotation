// imageViewer.js - Image Viewer Component (no zoom/pan yet)

export class ImageViewer {
    /**
     * @param {string|HTMLElement} container - CSS selector or DOM element to append the canvas to
     * @param {string} overlayId - ID of the loading overlay element
     * @param {string} [canvasId] - Optional canvas id to use or create
     */
    constructor(container, overlayId, canvasId = 'c') {
        // Resolve container
        if (typeof container === 'string') {
            this.container = document.querySelector(container);
        } else {
            this.container = container;
        }
        if (!this.container) {
            throw new Error('ImageViewer: container not found');
        }

        // Try to get existing canvas
        this.canvas = canvasId ? this.container.querySelector(`#${canvasId}`) : null;
        if (!this.canvas) {
            this.canvas = document.createElement('canvas');
            this.canvas.id = canvasId;
            this.container.appendChild(this.canvas);
        }
        this.ctx = this.canvas.getContext('2d');
        this.img = new window.Image();
        this.overlay = document.getElementById(overlayId);
        this.currentImageId = null;
        this.isLoading = false;
        this.img.onload = () => this.drawImageToCanvas();
    }

    setLoading(loading) {
        this.isLoading = loading;
        if (this.overlay) this.overlay.style.display = loading ? 'flex' : 'none';
    }

    async loadImage(imageUrl, imageId = null) {
        this.setLoading(true);
        this.currentImageId = imageId;
        this.img.src = imageUrl;
    }

    drawImageToCanvas() {
        // Fit image to canvas size
        const maxWidth = this.canvas.width;
        const maxHeight = this.canvas.height;
        this.ctx.clearRect(0, 0, maxWidth, maxHeight);
        // Center and fit image
        let drawWidth = this.img.width;
        let drawHeight = this.img.height;
        let scale = Math.min(maxWidth / drawWidth, maxHeight / drawHeight);
        let x = (maxWidth - drawWidth * scale) / 2;
        let y = (maxHeight - drawHeight * scale) / 2;
        this.ctx.drawImage(this.img, x, y, drawWidth * scale, drawHeight * scale);
        this.setLoading(false);
    }
}
