export class ImageView {
    constructor(container, overlayId, canvasId = 'c') {
        if (typeof container === 'string') {
            this.container = document.querySelector(container);
        } else {
            this.container = container;
        }
        if (!this.container) {
            throw new Error('ImageView: container not found');
        }
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
        this._resizeCanvasToContainer();
        this.img.onload = () => {
            this._resizeCanvasToContainer();
            this._drawImageToCanvas();
        };
        window.addEventListener('resize', () => {
            this._resizeCanvasToContainer();
            this._drawImageToCanvas();
        });
    }
    _resizeCanvasToContainer() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width * 0.9;
        this.canvas.height = rect.height * 0.9;
    }
    _setLoading(loading) {
        this.isLoading = loading;
        if (this.overlay) this.overlay.style.display = loading ? 'flex' : 'none';
    }

    _drawImageToCanvas() {
        const maxWidth = this.canvas.width;
        const maxHeight = this.canvas.height;
        this.ctx.clearRect(0, 0, maxWidth, maxHeight);
        let drawWidth = this.img.width;
        let drawHeight = this.img.height;
        let scale = Math.min(maxWidth / drawWidth, maxHeight / drawHeight);
        let x = (maxWidth - drawWidth * scale) / 2;
        let y = (maxHeight - drawHeight * scale) / 2;
        this.ctx.drawImage(this.img, x, y, drawWidth * scale, drawHeight * scale);
        this._setLoading(false);
    }

    async loadImage(imageUrl, imageId = null) {
        this._setLoading(true);
        this.currentImageId = imageId;
        if (this.previousObjectUrl) {
            URL.revokeObjectURL(this.previousObjectUrl);
        }
        this.previousObjectUrl = imageUrl;
        this.img.src = imageUrl;
    }
}
