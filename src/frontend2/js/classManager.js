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
    this.currentImageFilename = null;
    this.onClassChange = null; // callback(filename, className)
    this.onClassesUpdate = null; // callback(classes[])
    this.loadNextImage = loadNextImage;
    this.api = api;

        this.render();

        // Load any persisted classes from the backend
        this.loadClassesFromConfig();

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
            if (idx >= 0 && idx < this.globalClasses.length) {
                const className = this.globalClasses[idx];
                this.selectedClass = className;
                // POST to /annotate using API
                if (this.currentImageFilename && className) {
                    this.api.annotateSample(this.currentImageFilename, className)
                        .then(() => {
                            console.log('Annotation succeeded, calling loadNextImage (keyboard)');
                            if (typeof this.loadNextImage === 'function') {
                                this.loadNextImage();
                            }
                            else {
                                console.warn('loadNextImage callback is not defined');
                            }
                        })
                        .catch(err => {
                            console.error('Annotation request error:', err);
                        });
                }
                if (this.onClassChange) this.onClassChange(this.currentImageFilename, className);
                this.render();
            }
        });
    }

    // Load class list from server config
    async loadClassesFromConfig() {
        if (!this.api || typeof this.api.getConfig !== 'function') return;
        try {
            const cfg = await this.api.getConfig();
            if (cfg && Array.isArray(cfg.classes)) {
                this.globalClasses = cfg.classes;
            }
        } catch (e) {
            console.error('Failed to load classes from config:', e);
        }
    }

    // Setters for state
    async setCurrentImageFilename(filename, selectedClass = null) {
        this.currentImageFilename = filename;
        this.selectedClass = selectedClass;
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
                btn.className = 'class-btn' + (c === selected ? ' selected' : '');
                btn.dataset.class = c;
                btn.textContent = index < 10 ? `${c} (${index === 9 ? '0' : index + 1})` : c;
                btn.onclick = () => {
                    if (!this.currentImageFilename) return;
                    // Select class
                    this.selectedClass = c;
                    // POST to /annotate using API
                    this.api.annotateSample(this.currentImageFilename, c)
                        .then(() => {
                            console.log('Annotation succeeded, calling loadNextImage (button)');
                            if (typeof this.loadNextImage === 'function') {
                                this.loadNextImage();
                            } else {
                                console.warn('loadNextImage callback is not defined');
                            }
                        })
                        .catch(err => {
                            console.error('Annotation request error:', err);
                        });
                    if (this.onClassChange) this.onClassChange(this.currentImageFilename, c);
                    this.render();
                };

                // Remove button
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '✕';
                removeBtn.title = 'Remove class';
                removeBtn.style.marginLeft = '8px';
                removeBtn.style.background = '#dc3545';
                removeBtn.style.color = 'white';
                removeBtn.style.border = 'none';
                removeBtn.style.borderRadius = '4px';
                removeBtn.style.cursor = 'pointer';
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
