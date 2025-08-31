import { ClassesView } from '/shared/views/classesView.js';

export class PointsClassesView extends ClassesView {
    constructor(container, annotateWorkflow, state) {
        super(container, annotateWorkflow, state);
    }

    // Compute a distinct color per class index, stable across renders.
    generateClassColor(index) {
        const colors = [
            '#0080FF',
            '#FFFF80',
            '#000080',
            '#008080',
            '#00FF00',
            '#800000',
            '#80FFFF',
            '#0000FF',
            '#808000',
            '#8080FF',
            '#FF0080',
            '#008000',
            '#FF8000',
            '#FFFF00',
            '#80FF00',
            '#FF00FF',
            '#FFFFFF',
            '#00FF80',
            '#00FFFF',
            '#80FF80',
            '#FF0000',
            '#808080',
            '#8000FF',
            '#FF8080',
            '#000000',
            '#FF80FF',
            '#800080',
        ];
        return colors[index % colors.length];
    }

    // Get color for a class. Uses state.classColors Map if available to keep assignments stable.
    getClassColor(className) {
        const classes = (this.state && this.state.config && Array.isArray(this.state.config.classes))
            ? this.state.config.classes
            : [];
        const idx = Math.max(0, classes.indexOf(className));
        const colorFromIdx = this.generateClassColor(idx);

        // Prefer shared Map if provided in state to keep colors consistent across views.
        if (this.state && this.state.classColors instanceof Map) {
            if (!this.state.classColors.has(className)) {
                this.state.classColors.set(className, colorFromIdx);
            }
            return this.state.classColors.get(className);
        }
        return colorFromIdx;
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
            if (!className) return;
            if (!/^[A-Za-z0-9_-]{1,64}$/.test(className)) {
                alert('Invalid class name. Use letters, digits, underscore or hyphen (max 64).');
                return;
            }
            this.addClass(className);
            input.value = '';
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
                
                // Color indicator
                const colorIndicator = document.createElement('div');
                const classColor = this.getClassColor(c);
                colorIndicator.style.width = '12px';
                colorIndicator.style.height = '12px';
                colorIndicator.style.backgroundColor = classColor;
                colorIndicator.style.border = '1px solid #ccc';
                colorIndicator.style.marginRight = '6px';
                colorIndicator.style.borderRadius = '2px';
                colorIndicator.style.flexShrink = '0';

                const btn = document.createElement('button');
                const isSelected = c === selected;
                const isPrediction = !isSelected && c === this.predictionClass;
                btn.className = 'btn class-btn' + (isSelected ? ' selected' : isPrediction ? ' prediction' : '');
                btn.dataset.class = c;
                btn.textContent = index < 10 ? `${c} (${index === 9 ? '0' : index + 1})` : c;
                btn.disabled = this.isLoading;
                
                // Add subtle background color to selected button
                if (isSelected) {
                    btn.style.backgroundColor = classColor + '40'; // 25% opacity
                    btn.style.borderColor = classColor;
                }

                btn.onclick = async () => {
                    if (!this.currentSampleId || this.isLoading) return;
                    const sampleId = this.currentSampleId;
                    this.selectedClass = c;
                    if (this.onClassChange) this.onClassChange(sampleId, c);
                    
                    // For points annotation, we just select the class instead of annotating immediately
                    // The actual annotation happens when clicking on the image
                    try {
                        await this.annotateWorkflow(sampleId, c); // This will be selectClassWorkflow
                        this.renderUI(); // Re-render to show selection
                        console.log('Class selected:', c);
                    } catch (err) {
                        console.error('Class selection error:', err);
                    }
                };

                const removeBtn = document.createElement('button');
                removeBtn.textContent = '✕';
                removeBtn.title = 'Remove class';
                removeBtn.className = 'btn remove-btn';
                removeBtn.onclick = () => {
                    this.removeClass(c);
                };

                classRow.appendChild(colorIndicator);
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
