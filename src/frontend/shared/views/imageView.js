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
        
        // Optional mask overlays and alpha (0..1)
        this._overlayAlpha = 0; // default off
        this._maskOverlays = null; // { [className]: HTMLImageElement }
        this._maskOverlayColors = null; // { [className]: "#RRGGBB" }
        
        // Points annotation state
        this.points = []; // Array to store added points {x, y, className, color}
        this.onPointAdd = null; // Callback for when a point is added
        this.onPointRemove = null; // Callback for when a point is removed
        this.selectedClass = null;
        this.selectedClassColor = null;
        this.hoveredPointIndex = -1; // Index of point being hovered over
        
        // Image transformation state
        this.imageTransform = {
            x: 0, y: 0, width: 0, height: 0, scale: 1
        };
        
        this._resizeCanvasToContainer();
        this._setupEventListeners();
        this.img.onload = () => {
            this._resizeCanvasToContainer();
            this._drawImageToCanvas();
        };
        window.addEventListener('resize', () => {
            this._resizeCanvasToContainer();
            this._drawImageToCanvas();
        });
    }
    
    _setupEventListeners() {
        // Handle left click to add points
        this.canvas.addEventListener('click', (e) => {
            if (e.button !== 0 || this.isLoading || !this.selectedClass) return; // Only left click
            
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            // Convert canvas coordinates to image coordinates
            const imageCoords = this._canvasToImageCoords(canvasX, canvasY);
            if (imageCoords) {
                this._addPoint(imageCoords.x, imageCoords.y, this.selectedClass, this.selectedClassColor);
            }
        });
        
        // Handle right click to remove points
        this.canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault(); // Prevent context menu
            
            if (this.isLoading) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            // Convert canvas coordinates to image coordinates
            const imageCoords = this._canvasToImageCoords(canvasX, canvasY);
            if (imageCoords) {
                this._removePointNear(imageCoords.x, imageCoords.y);
            }
        });
        
        // Add hover effect for cursor
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.selectedClass) {
                this.canvas.classList.add('crosshair');
            } else {
                this.canvas.classList.remove('crosshair');
            }
            
            // Check if hovering near a point for removal feedback
            const rect = this.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            const imageCoords = this._canvasToImageCoords(canvasX, canvasY);
            
            if (imageCoords) {
                this._updateHoverState(imageCoords.x, imageCoords.y);
            }
        });
    }
    
    _canvasToImageCoords(canvasX, canvasY) {
        if (!this.imageTransform.width || !this.imageTransform.height) return null;
        
        // Check if click is within the image bounds
        const imgLeft = this.imageTransform.x;
        const imgTop = this.imageTransform.y;
        const imgRight = imgLeft + this.imageTransform.width;
        const imgBottom = imgTop + this.imageTransform.height;
        
        if (canvasX < imgLeft || canvasX > imgRight || canvasY < imgTop || canvasY > imgBottom) {
            return null; // Click outside image
        }
        
        // Convert to image coordinates (normalized 0-1)
        const relativeX = (canvasX - imgLeft) / this.imageTransform.width;
        const relativeY = (canvasY - imgTop) / this.imageTransform.height;
        
        return {
            x: relativeX,
            y: relativeY
        };
    }
    
    _addPoint(x, y, className, color) {
        const point = { x, y, className, color };
        this.points.push(point);
        this._drawImageToCanvas(); // Redraw to include the new point
        
        if (this.onPointAdd) {
            this.onPointAdd(point);
        }
    }
    
    addExistingPoint(x, y, className, color) {
        // Add a point without triggering the onPointAdd callback (for loading existing annotations)
        const point = { x, y, className, color };
        this.points.push(point);
        this._drawImageToCanvas(); // Redraw to include the point
    }
    
    _removePointNear(x, y) {
        const threshold = 0.02; // 2% of image size for click tolerance
        let closestIndex = -1;
        let closestDistance = Infinity;
        
        // Find the closest point to the click
        this.points.forEach((point, index) => {
            const dx = point.x - x;
            const dy = point.y - y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < threshold && distance < closestDistance) {
                closestDistance = distance;
                closestIndex = index;
            }
        });
        
        // Remove the closest point if found
        if (closestIndex !== -1) {
            const removedPoint = this.points.splice(closestIndex, 1)[0];
            this._drawImageToCanvas(); // Redraw without the removed point
            
            if (this.onPointRemove) {
                this.onPointRemove(removedPoint, closestIndex);
            }
            
            console.log('Removed point:', removedPoint);
        }
    }
    
    _updateHoverState(x, y) {
        const threshold = 0.02; // 2% of image size for hover detection
        let closestIndex = -1;
        let closestDistance = Infinity;
        
        // Find the closest point to the mouse
        this.points.forEach((point, index) => {
            const dx = point.x - x;
            const dy = point.y - y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < threshold && distance < closestDistance) {
                closestDistance = distance;
                closestIndex = index;
            }
        });
        
        // Update hover state if changed
        if (this.hoveredPointIndex !== closestIndex) {
            this.hoveredPointIndex = closestIndex;
            this._drawImageToCanvas(); // Redraw to show hover effect
        }
    }
    
    setSelectedClass(className, color) {
        this.selectedClass = className;
        this.selectedClassColor = color;
        
        // Update cursor immediately
        if (className) {
            this.canvas.classList.add('crosshair');
        } else {
            this.canvas.classList.remove('crosshair');
        }
    }
    
    clearPoints() {
        this.points = [];
        this.hoveredPointIndex = -1;
        this._drawImageToCanvas();
    }
    
    removeLastPoint() {
        if (this.points.length > 0) {
            this.points.pop();
            this.hoveredPointIndex = -1;
            this._drawImageToCanvas();
        }
    }
    
    getLastPoint() {
        return this.points.length > 0 ? this.points[this.points.length - 1] : null;
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
        
        if (!this.img.complete || !this.img.naturalWidth) return;
        
        let drawWidth = this.img.width;
        let drawHeight = this.img.height;
        let scale = Math.min(maxWidth / drawWidth, maxHeight / drawHeight);
        let x = (maxWidth - drawWidth * scale) / 2;
        let y = (maxHeight - drawHeight * scale) / 2;
        
        // Store image transformation for coordinate conversion
        this.imageTransform = {
            x: x,
            y: y,
            width: drawWidth * scale,
            height: drawHeight * scale,
            scale: scale
        };
        
        this.ctx.drawImage(this.img, x, y, drawWidth * scale, drawHeight * scale);
        // Draw mask overlays (if any) scaled to image bounds, then points
        this._drawMaskOverlays();
        
        // Draw points on top of the image
        this._drawPoints();
        
        this._setLoading(false);
    }

    // Draw any provided mask overlays, scaled to the displayed image bounds
    _drawMaskOverlays() {
        if (!this._maskOverlays || !this.imageTransform.width) return;
        const alpha = Math.max(0, Math.min(1, Number(this._overlayAlpha || 0)));
        if (alpha <= 0) return;
        const { x, y, width, height } = this.imageTransform;
        
        this.ctx.save();
        this.ctx.globalAlpha = alpha;
        // Support either a single Image or a map of className -> Image
        const overlays = this._maskOverlays;
        if (overlays instanceof window.Image) {
            if (overlays.complete && overlays.naturalWidth) {
                this.ctx.drawImage(overlays, x, y, width, height);
            }
        } else if (typeof overlays === 'object') {
            for (const key of Object.keys(overlays)) {
                let img = overlays[key];
                let tintColor = null;
                if (img && !(img instanceof window.Image) && typeof img === 'object') {
                    // Accept structures like { image: Image, color: string } or { img: Image, color }
                    if (img.image instanceof window.Image) {
                        tintColor = img.color || null;
                        img = img.image;
                    } else if (img.img instanceof window.Image) {
                        tintColor = img.color || null;
                        img = img.img;
                    }
                }
                if (!tintColor && this._maskOverlayColors && this._maskOverlayColors[key]) {
                    tintColor = this._maskOverlayColors[key];
                }
                if (img && img.complete && img.naturalWidth) {
                    if (tintColor) {
                        // Draw tinted using offscreen canvas and source-in composite
                        const off = document.createElement('canvas');
                        off.width = Math.max(1, Math.floor(width));
                        off.height = Math.max(1, Math.floor(height));
                        const offCtx = off.getContext('2d');
                        // Draw the mask scaled to the offscreen
                        offCtx.clearRect(0, 0, off.width, off.height);
                        offCtx.drawImage(img, 0, 0, off.width, off.height);
                        // Use the existing alpha of the mask to clip the fill
                        offCtx.globalCompositeOperation = 'source-in';
                        offCtx.fillStyle = tintColor;
                        offCtx.fillRect(0, 0, off.width, off.height);
                        // Paint onto main canvas at image bounds
                        this.ctx.drawImage(off, x, y, width, height);
                    } else {
                        // No tint provided; draw raw mask
                        this.ctx.drawImage(img, x, y, width, height);
                    }
                }
            }
        }
        this.ctx.restore();
    }

    // Overlay alpha accessor triggers redraw
    get overlayAlpha() { return this._overlayAlpha; }
    set overlayAlpha(a) {
        const v = Math.max(0, Math.min(1, Number(a || 0)));
        if (v === this._overlayAlpha) return;
        this._overlayAlpha = v;
        this._drawImageToCanvas();
    }

    // Mask overlays accessor triggers redraw
    get maskOverlays() { return this._maskOverlays; }
    set maskOverlays(val) {
        this._maskOverlays = val || null;
        this._drawImageToCanvas();
    }

    // Optional per-class colors for mask overlays
    get maskOverlayColors() { return this._maskOverlayColors; }
    set maskOverlayColors(val) {
        this._maskOverlayColors = val || null;
        this._drawImageToCanvas();
    }
    
    _drawPoints() {
        if (!this.points.length || !this.imageTransform.width) return;
        
        this.points.forEach((point, index) => {
            // Convert normalized coordinates back to canvas coordinates
            const canvasX = this.imageTransform.x + (point.x * this.imageTransform.width);
            const canvasY = this.imageTransform.y + (point.y * this.imageTransform.height);
            
            // Determine if this point is hovered
            const isHovered = index === this.hoveredPointIndex;
            const radius = isHovered ? 8 : 6; // Larger radius when hovered
            
            // Draw point
            this.ctx.fillStyle = point.color || '#FF0000';
            this.ctx.strokeStyle = isHovered ? '#FFD700' : '#FFFFFF'; // Gold border when hovered
            this.ctx.lineWidth = isHovered ? 3 : 2;
            
            this.ctx.beginPath();
            this.ctx.arc(canvasX, canvasY, radius, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.stroke();
            
            // Draw class label
            if (point.className) {
                this.ctx.fillStyle = isHovered ? '#FFD700' : '#000000';
                this.ctx.font = isHovered ? 'bold 12px Arial' : '12px Arial';
                this.ctx.textAlign = 'center';
                this.ctx.fillText(point.className, canvasX, canvasY - (radius + 4));
            }
        });
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
