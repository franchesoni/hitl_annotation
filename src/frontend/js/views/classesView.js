// classesView.js - Class Manager Component

/**
 * Handles class input, display, and selection for annotation UI.
 * Usage: new ClassesView(containerElementOrSelector)
 *
 * Keyboard shortcuts: numbers 1-9,0 select the corresponding class button (if present).
 */
export class ClassesView {
    /**
     * @param {string|HTMLElement} container - CSS selector or DOM element for the class list UI
     * @param {function} annotateWorkflow - callback to handle annotation workflow
     * @param {object} state - reference to main app state (will be modified directly)
     */
    constructor(container, annotateWorkflow, state) {
        // Find container element
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) throw new Error('ClassesView: container not found');

    // State - reference to main app state
    this.state = state; // Direct reference, changes will be reflected in main app
    this.selectedClass = null;
    this.predictionClass = null;
    this.currentSampleId = null;
    this.currentImageFilename = null; // Keep for display purposes
    this.onClassChange = null; // callback(sampleId, className)
    this.onClassesUpdate = null; // callback(classes[])
    this.annotateWorkflow = annotateWorkflow;
    this.isLoading = false;
    this.annotationRequestId = 0;

        this.render();

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
                if (idx >= 0 && idx < this.state.config.classes.length) {
                const className = this.state.config.classes[idx];
                const sampleId = this.currentSampleId; // capture sample ID before async ops
                this.selectedClass = className;                // Record the annotation before starting async operations so undo works immediately
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

    // Render with current state (no parameters needed)
    render() {
        this.renderUI();
    }

    // Setters for state
    async setCurrentSample(sampleId, filepath, selectedClass = null) {
        this.currentSampleId = sampleId;
        this.currentImageFilename = filepath; // Keep for display
        if (selectedClass !== null) {
            this.selectedClass = selectedClass;
        }
        this.renderUI();
    }
    
    // Keep for backward compatibility
    async setCurrentImageFilename(filename, selectedClass = null) {
        this.currentImageFilename = filename;
        if (selectedClass !== null) {
            this.selectedClass = selectedClass;
        }
        this.renderUI();
    }
    setSelectedClass(className) {
        this.selectedClass = className;
        this.renderUI();
    }
    setGlobalClasses(classes) {
        this.state.config.classes = [...classes];
        this.state.configUpdated = true;
        this.renderUI();
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
        this.renderUI();
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
        if (!className || this.state.config.classes.includes(className)) return;
        this.state.config.classes.push(className);
        this.state.configUpdated = true;
        this.renderUI();
    }

    // Remove a class by name
    async removeClass(className) {
        this.state.config.classes = this.state.config.classes.filter(c => c !== className);
        this.state.configUpdated = true;
        this.renderUI();
    }

    // Render the class manager UI
    renderUI() {
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
        if (this.state.config.classes.length === 0) {
            // No classes yet
            classListDiv.innerHTML = '<div style="color: #6c757d; font-style: italic;">No classes added yet</div>';
        } else {
            // Create a button for each class
            const selected = this.selectedClass;
            this.state.config.classes.forEach((c, index) => {
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
                    this.renderUI();
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
            this.onClassesUpdate([...this.state.config.classes]);
        }
    }
}
