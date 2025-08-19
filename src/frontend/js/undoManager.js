export class UndoManager {
    constructor(api, viewer, classManager, updatePrediction) {
        this.api = api;
        this.viewer = viewer;
        this.classManager = classManager;
        this.updatePrediction = updatePrediction;
        this.history = [];

        // Keyboard shortcuts for undo
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const key = e.key.toLowerCase();
            if ((e.ctrlKey || e.metaKey) && key === 'z') {
                e.preventDefault();
                this.undo();
            } else if (e.key === 'Backspace' || key === 'u') {
                e.preventDefault();
                this.undo();
            }
        });
    }

    record(sampleId) {
        if (sampleId) {
            this.history.push(sampleId);
        }
    }

    async undo() {
        if (this.history.length === 0) {
            alert('No more actions to undo');
            return;
        }
        const sampleId = this.history.pop();
        try {
            await this.api.deleteAnnotation(sampleId);
        } catch (e) {
            console.error('Failed to delete annotation:', e);
        }
        try {
            const { imageUrl, sampleId: returnedSampleId, filepath, labelClass, labelSource, labelProbability } = await this.api.loadSample(sampleId);
            this.viewer.loadImage(imageUrl, filepath);
            const cls = labelSource === 'annotation' ? labelClass : null;
            await this.classManager.setCurrentSample(returnedSampleId, filepath, cls);
            if (this.updatePrediction) {
                this.updatePrediction(labelClass, labelProbability, labelSource);
            }
        } catch (e) {
            console.error('Failed to load sample:', e);
        }
    }
}
