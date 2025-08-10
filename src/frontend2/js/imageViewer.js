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
        this.previousObjectUrl = null;

        // Initial sizing
        this.resizeCanvasToContainer();

        // Redraw and resize on image load
        this.img.onload = () => {
            this.resizeCanvasToContainer();
            this.drawImageToCanvas();
        };

        // Redraw and resize on window resize
        window.addEventListener('resize', () => {
            this.resizeCanvasToContainer();
            this.drawImageToCanvas();
        });
    }

    resizeCanvasToContainer() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width * 0.9;
        this.canvas.height = rect.height * 0.9;
    }

    setLoading(loading) {
        this.isLoading = loading;
        if (this.overlay) this.overlay.style.display = loading ? 'flex' : 'none';
    }

    async loadImage(imageUrl, imageId = null) {
        this.setLoading(true);
        this.currentImageId = imageId;
        if (this.previousObjectUrl) {
            URL.revokeObjectURL(this.previousObjectUrl);
        }
        this.previousObjectUrl = imageUrl;
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
