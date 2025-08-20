export class ClassesView {
    constructor(container, annotateWorkflow, state) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) throw new Error('ClassesView: container not found');
    this.state = state; 
    this.selectedClass = null;
    this.predictionClass = null;
    this.currentSampleId = null;
    this.currentImageFilename = null; 
    this.onClassChange = null; 
    this.onClassesUpdate = null; 
    this.annotateWorkflow = annotateWorkflow;
    this.isLoading = false;
    this.annotationRequestId = 0;
        this.render();
        document.addEventListener('keydown', async (e) => {
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
                const sampleId = this.currentSampleId; 
                this.selectedClass = className;                
                if (this.onClassChange) this.onClassChange(sampleId, className);
                this.render();
                if (sampleId && className) {
                    let requestId;
                    try {
                        this.setLoading(true);
                        requestId = ++this.annotationRequestId; 
                        await this.annotateWorkflow(sampleId, className);
                        if (requestId !== this.annotationRequestId) return; 
                        if (this.onAnnotationSuccess) this.onAnnotationSuccess(sampleId, className);
                        console.log('Annotation workflow completed (keyboard)');
                    } catch (err) {
                        console.error('Annotation workflow error:', err);
                    } finally {
                        if (this.annotationRequestId === requestId) this.setLoading(false);
                    }
                }
            }
        });
    }
    render() {
        this.renderUI();
    }
    async setCurrentSample(sampleId, filepath, selectedClass = null) {
        this.currentSampleId = sampleId;
        this.currentImageFilename = filepath; 
        if (selectedClass !== null) {
            this.selectedClass = selectedClass;
        }
        this.renderUI();
    }
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
    async addClass(className) {
        if (!className || this.state.config.classes.includes(className)) return;
        this.state.config.classes.push(className);
        this.state.configUpdated = true;
        this.renderUI();
    }
    async removeClass(className) {
        this.state.config.classes = this.state.config.classes.filter(c => c !== className);
        this.state.configUpdated = true;
        this.renderUI();
    }
    renderUI() {
        this.container.innerHTML = '';
        const inputGroup = document.createElement('div');
        inputGroup.className = 'class-input-group';
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = 'Add new class...';
        const addBtn = document.createElement('button');
        addBtn.className = 'btn';
        addBtn.textContent = 'Add (Enter)';
        inputGroup.appendChild(input);
        inputGroup.appendChild(addBtn);
        this.container.appendChild(inputGroup);
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
        const classListDiv = document.createElement('div');
        classListDiv.className = 'class-buttons';
        if (this.state.config.classes.length === 0) {
            classListDiv.innerHTML = '<div style="color: #6c757d; font-style: italic;">No classes added yet</div>';
        } else {
            const selected = this.selectedClass;
            this.state.config.classes.forEach((c, index) => {
                const classRow = document.createElement('div');
                classRow.style.display = 'flex';
                classRow.style.alignItems = 'center';
                const btn = document.createElement('button');
                const isSelected = c === selected;
                const isPrediction = !isSelected && c === this.predictionClass;
                btn.className = 'btn class-btn' + (isSelected ? ' selected' : isPrediction ? ' prediction' : '');
                btn.dataset.class = c;
                btn.textContent = index < 10 ? `${c} (${index === 9 ? '0' : index + 1})` : c;
                btn.disabled = this.isLoading;
                btn.onclick = async () => {
                    if (!this.currentSampleId || this.isLoading) return;
                    const sampleId = this.currentSampleId; 
                    this.selectedClass = c;
                    if (this.onClassChange) this.onClassChange(sampleId, c);
                    this.renderUI();
                    let requestId;
                    try {
                        this.setLoading(true);
                        requestId = ++this.annotationRequestId; 
                        await this.annotateWorkflow(sampleId, c);
                        if (requestId !== this.annotationRequestId) return; 
                        if (this.onAnnotationSuccess) this.onAnnotationSuccess(sampleId, c);
                        console.log('Annotation workflow completed (button)');
                    } catch (err) {
                        console.error('Annotation workflow error:', err);
                    } finally {
                        if (this.annotationRequestId === requestId) this.setLoading(false);
                    }
                };
                const removeBtn = document.createElement('button');
                removeBtn.textContent = '✕';
                removeBtn.title = 'Remove class';
                removeBtn.className = 'btn remove-btn';
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
