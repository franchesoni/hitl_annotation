export class UndoManager {
    constructor(api, viewer, classManager, updatePrediction) {
        this.api = api;
        this.viewer = viewer;
        this.classManager = classManager;
        this.updatePrediction = updatePrediction;
        this.history = [];

        // Keyboard shortcuts for undo: Backspace or 'u'
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            if (e.key === 'Backspace' || e.key.toLowerCase() === 'u') {
                e.preventDefault();
                this.undo();
            }
        });
    }

    record(imageId) {
        if (imageId) {
            this.history.push(imageId);
        }
    }

    async undo() {
        if (this.history.length === 0) {
            alert('No more actions to undo');
            return;
        }
        const id = this.history.pop();
        try {
            await this.api.deleteAnnotation(id);
        } catch (e) {
            console.error('Failed to delete annotation:', e);
        }
        try {
            const { imageUrl, filename, labelClass, labelSource, labelProbability } = await this.api.loadSample(id);
            this.viewer.loadImage(imageUrl, filename);
            const cls = labelSource === 'annotation' ? labelClass : null;
            await this.classManager.setCurrentImageFilename(filename, cls);
            if (this.updatePrediction) {
                this.updatePrediction(labelClass, labelProbability, labelSource);
            }
        } catch (e) {
            console.error('Failed to load sample:', e);
        }
    }
}
