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
     * @param {function} annotateWorkflow - callback to handle annotation workflow
     */
    constructor(container, annotateWorkflow, api) {
        // Find container element
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) throw new Error('ClassManager: container not found');

    // State
    this.globalClasses = [];
    this.selectedClass = null;
    this.predictionClass = null;
    this.currentSampleId = null;
    this.currentImageFilename = null; // Keep for display purposes
    this.onClassChange = null; // callback(sampleId, className)
    this.onClassesUpdate = null; // callback(classes[])
    this.annotateWorkflow = annotateWorkflow;
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
                const sampleId = this.currentSampleId; // capture sample ID before async ops
                this.selectedClass = className;

                // Record the annotation before starting async operations so undo works immediately
                if (this.onClassChange) this.onClassChange(sampleId, className);
                this.render();

                if (sampleId && className) {
                    let requestId;
                    try {
                        this.setLoading(true);
                        requestId = ++this.annotationRequestId; // store in outer scope for finally access
                        
                        // Use the annotation workflow instead of individual API calls
                        await this.annotateWorkflow(sampleId, className);
                        if (requestId !== this.annotationRequestId) return; // stale response

                        // Call success callback after successful annotation
                        if (this.onAnnotationSuccess) this.onAnnotationSuccess(sampleId, className);

                        console.log('Annotation workflow completed (keyboard)');
                    } catch (err) {
                        console.error('Annotation workflow error:', err);
                    } finally {
                        // Only clear loading if this request is still the latest
                        if (this.annotationRequestId === requestId) this.setLoading(false);
                    }
                }
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
    async setCurrentSample(sampleId, filepath, selectedClass = null) {
        this.currentSampleId = sampleId;
        this.currentImageFilename = filepath; // Keep for display
        if (selectedClass !== null) {
            this.selectedClass = selectedClass;
        }
        await this.loadClassesFromConfig();
        this.render();
    }
    
    // Keep for backward compatibility
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

    setOnAnnotationSuccess(callback) {
        this.onAnnotationSuccess = callback;
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

    // Add a new class if not already present (frontend only, will be synced during annotation)
    async addClass(className) {
        if (!className || this.globalClasses.includes(className)) return;
        this.globalClasses.push(className);
        // Don't push to backend immediately - will be synced during annotation/undo workflows
        this.render();
    }

    // Remove a class by name (frontend only, will be synced during annotation)
    async removeClass(className) {
        this.globalClasses = this.globalClasses.filter(c => c !== className);
        // Don't push to backend immediately - will be synced during annotation/undo workflows
        this.render();
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
                    if (!this.currentSampleId || this.isLoading) return;
                    const sampleId = this.currentSampleId; // capture current sample ID
                    this.selectedClass = c;
                    if (this.onClassChange) this.onClassChange(sampleId, c);
                    this.render();
                    let requestId;
                    try {
                        this.setLoading(true);
                        requestId = ++this.annotationRequestId; // make visible to finally
                        
                        // Use the annotation workflow instead of individual API calls
                        await this.annotateWorkflow(sampleId, c);
                        if (requestId !== this.annotationRequestId) return; // stale response
                        
                        // Call success callback after successful annotation
                        if (this.onAnnotationSuccess) this.onAnnotationSuccess(sampleId, c);
                        
                        console.log('Annotation workflow completed (button)');
                    } catch (err) {
                        console.error('Annotation workflow error:', err);
                    } finally {
                        if (this.annotationRequestId === requestId) this.setLoading(false);
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
