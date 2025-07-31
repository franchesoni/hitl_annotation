// classManager.js - Class Manager Component

/**
 * Handles class input, display, and selection for annotation UI.
 * Usage: new ClassManager(containerElementOrSelector)
 *
 * Keyboard shortcuts: numbers 1-9,0 select the corresponding class button (if present).
 */
export class ClassManager {
    // Keyboard shortcuts are not handled inside this component by default.
    // You must add a keydown event listener in your main app and call the appropriate ClassManager method.
    /**
     * @param {string|HTMLElement} container - CSS selector or DOM element for the class list UI
     */
    constructor(container) {
        // Find container element
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) throw new Error('ClassManager: container not found');

        // State
        this.globalClasses = [];
        this.imageSelectedClass = {};
        this.currentImageId = null;
        this.onClassChange = null; // callback(imageId, className)

        this.render();

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
                this.imageSelectedClass[this.currentImageId] = className;
                if (this.onClassChange) this.onClassChange(this.currentImageId, className);
                this.render();
            }
        });
    }

    // Setters for state
    setCurrentImageId(imageId) {
        this.currentImageId = imageId;
        this.render();
    }
    setGlobalClasses(classes) {
        this.globalClasses = classes;
        this.render();
    }
    setImageSelectedClass(imageSelectedClass) {
        this.imageSelectedClass = imageSelectedClass;
        this.render();
    }
    setOnClassChange(callback) {
        this.onClassChange = callback;
    }

    // Add a new class if not already present
    addClass(className) {
        if (!className || this.globalClasses.includes(className)) return;
        this.globalClasses.push(className);
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
            const selected = this.currentImageId ? this.imageSelectedClass[this.currentImageId] : undefined;
            this.globalClasses.forEach((c, index) => {
                const btn = document.createElement('button');
                btn.className = 'class-btn' + (c === selected ? ' selected' : '');
                btn.dataset.class = c;
                btn.textContent = index < 10 ? `${c} (${index === 9 ? '0' : index + 1})` : c;
                btn.onclick = () => {
                    if (!this.currentImageId) return;
                    // Toggle selection
                    if (this.imageSelectedClass[this.currentImageId] === c) {
                        delete this.imageSelectedClass[this.currentImageId];
                        if (this.onClassChange) this.onClassChange(this.currentImageId, null);
                    } else {
                        this.imageSelectedClass[this.currentImageId] = c;
                        if (this.onClassChange) this.onClassChange(this.currentImageId, c);
                    }
                    this.render();
                };
                classListDiv.appendChild(btn);
            });
        }
        this.container.appendChild(classListDiv);
    }
}
