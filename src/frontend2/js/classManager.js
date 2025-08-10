// classManager.js - Class Manager Component

/**
 * Handles class input, display, and selection for annotation UI.
 * Usage: new ClassManager(containerElementOrSelector)
 *
 * Keyboard shortcuts: numbers 1-9,0 select the corresponding class button (if present).
 */
export class ClassManager {
    /**
     * @param {string|HTMLElement} container - CSS selector or DOM element for the class list UI
     * @param {function} loadNextImage - callback to load the next image
     */
    constructor(container, loadNextImage, api) {
        // Find container element
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) throw new Error('ClassManager: container not found');

    // State
    this.globalClasses = [];
    this.selectedClass = null;
    this.predictionClass = null;
    this.currentImageFilename = null;
    this.onClassChange = null; // callback(filename, className)
    this.onClassesUpdate = null; // callback(classes[])
    this.loadNextImage = loadNextImage;
    this.api = api;
    this.isLoading = false;
    this.annotationRequestId = 0;
    this.configRequestId = 0;

        this.render();

        // Load any persisted classes from the backend
        this.loadClassesFromConfig();

        // Keyboard shortcuts for class selection
        document.addEventListener('keydown', async (e) => {
            // Only trigger if not typing in an input/textarea or currently loading
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (this.isLoading) return;
            let idx = -1;
            if (e.key >= '1' && e.key <= '9') {
                idx = parseInt(e.key, 10) - 1;
            } else if (e.key === '0') {
                idx = 9;
            }
            if (idx >= 0 && idx < this.globalClasses.length) {
                const className = this.globalClasses[idx];
                this.selectedClass = className;
                if (this.currentImageFilename && className) {
                    try {
                        this.setLoading(true);
                        const requestId = ++this.annotationRequestId;
                        await this.api.annotateSample(this.currentImageFilename, className);
                        if (requestId !== this.annotationRequestId) return;
                        console.log('Annotation succeeded, calling loadNextImage (keyboard)');
                        if (typeof this.loadNextImage === 'function') {
                            await this.loadNextImage();
                        } else {
                            console.warn('loadNextImage callback is not defined');
                        }
                    } catch (err) {
                        console.error('Annotation request error:', err);
                    } finally {
                        if (this.annotationRequestId === requestId) {
                            this.setLoading(false);
                        }
                    }
                }
                if (this.onClassChange) this.onClassChange(this.currentImageFilename, className);
                this.render();
            }
        });
    }

    // Load class list from server config
    async loadClassesFromConfig() {
        if (!this.api || typeof this.api.getConfig !== 'function') return;
        const requestId = ++this.configRequestId;
        try {
            const cfg = await this.api.getConfig();
            if (requestId !== this.configRequestId) return;
            if (cfg && Array.isArray(cfg.classes)) {
                this.globalClasses = cfg.classes;
            }
        } catch (e) {
            if (requestId === this.configRequestId) {
                console.error('Failed to load classes from config:', e);
            }
        }
    }

    // Setters for state
    async setCurrentImageFilename(filename, selectedClass = null) {
        this.currentImageFilename = filename;
        if (selectedClass !== null) {
            this.selectedClass = selectedClass;
        }
        await this.loadClassesFromConfig();
        this.render();
    }
    setSelectedClass(className) {
        this.selectedClass = className;
        this.render();
    }
    setGlobalClasses(classes) {
        this.globalClasses = classes;
        this.render();
    }
    setOnClassChange(callback) {
        this.onClassChange = callback;
    }

    setOnClassesUpdate(callback) {
        this.onClassesUpdate = callback;
    }

    setPrediction(className) {
        this.predictionClass = className;
        this.render();
    }

    setLoading(isLoading) {
        this.isLoading = isLoading;
        const buttons = this.container.querySelectorAll('.class-btn');
        buttons.forEach(btn => {
            btn.disabled = isLoading;
        });
    }

    // Add a new class if not already present
    async addClass(className) {
        if (!className || this.globalClasses.includes(className)) return;
        this.globalClasses.push(className);
        await this.updateConfig();
        this.render();
    }

    // Remove a class by name
    async removeClass(className) {
        this.globalClasses = this.globalClasses.filter(c => c !== className);
        await this.updateConfig();
        this.render();
    }

    // Update backend config with current class list
    async updateConfig() {
        try {
            await fetch('/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ classes: this.globalClasses })
            });
        } catch (e) {
            console.error('Failed to update class list on server:', e);
        }
    }

    // Render the class manager UI
    render() {
        // Clear container
        this.container.innerHTML = '';

        // --- Input group for adding new class ---
        const inputGroup = document.createElement('div');
        inputGroup.className = 'class-input-group';
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = 'Add new class...';
        const addBtn = document.createElement('button');
        addBtn.textContent = 'Add (Enter)';
        inputGroup.appendChild(input);
        inputGroup.appendChild(addBtn);
        this.container.appendChild(inputGroup);

        // Add class on button click or Enter
        addBtn.onclick = () => {
            const className = input.value.trim();
            if (className) {
                this.addClass(className);
                input.value = '';
            }
        };
        input.onkeydown = (e) => {
            if (e.key === 'Enter') addBtn.click();
        };

        // --- Class buttons ---
        const classListDiv = document.createElement('div');
        classListDiv.className = 'class-buttons';
        if (this.globalClasses.length === 0) {
            // No classes yet
            classListDiv.innerHTML = '<div style="color: #6c757d; font-style: italic;">No classes added yet</div>';
        } else {
            // Create a button for each class
            const selected = this.selectedClass;
            this.globalClasses.forEach((c, index) => {
                const classRow = document.createElement('div');
                classRow.style.display = 'flex';
                classRow.style.alignItems = 'center';

                const btn = document.createElement('button');
                const isSelected = c === selected;
                const isPrediction = !isSelected && c === this.predictionClass;
                btn.className = 'class-btn' + (isSelected ? ' selected' : isPrediction ? ' prediction' : '');
                btn.dataset.class = c;
                btn.textContent = index < 10 ? `${c} (${index === 9 ? '0' : index + 1})` : c;
                btn.disabled = this.isLoading;
                btn.onclick = async () => {
                    if (!this.currentImageFilename || this.isLoading) return;
                    this.selectedClass = c;
                    if (this.onClassChange) this.onClassChange(this.currentImageFilename, c);
                    this.render();
                    try {
                        this.setLoading(true);
                        const requestId = ++this.annotationRequestId;
                        await this.api.annotateSample(this.currentImageFilename, c);
                        if (requestId !== this.annotationRequestId) return;
                        console.log('Annotation succeeded, calling loadNextImage (button)');
                        if (typeof this.loadNextImage === 'function') {
                            await this.loadNextImage();
                        } else {
                            console.warn('loadNextImage callback is not defined');
                        }
                    } catch (err) {
                        console.error('Annotation request error:', err);
                    } finally {
                        if (this.annotationRequestId === requestId) {
                            this.setLoading(false);
                        }
                    }
                };

                // Remove button
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '✕';
                removeBtn.title = 'Remove class';
                removeBtn.className = 'remove-btn';
                removeBtn.onclick = () => {
                    this.removeClass(c);
                };

                classRow.appendChild(btn);
                classRow.appendChild(removeBtn);
                classListDiv.appendChild(classRow);
            });
        }
        this.container.appendChild(classListDiv);
        if (this.onClassesUpdate) {
            this.onClassesUpdate([...this.globalClasses]);
        }
    }
}
