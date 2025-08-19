export class UndoManager {
    constructor(api, viewer, classManager, updatePrediction, updateStats, updateTrainingCurve) {
        this.api = api;
        this.viewer = viewer;
        this.classManager = classManager;
        this.updatePrediction = updatePrediction;
        this.updateStats = updateStats;
        this.updateTrainingCurve = updateTrainingCurve;
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
            // Undo workflow: post config → get prev → delete annotation → get stats → get config
            
            // Step 1: Post config (push frontend class changes to backend)
            await this.api.updateConfig({ classes: this.classManager.globalClasses });
            
            // Step 2: Get prev (load the sample we're undoing)
            const { imageUrl, sampleId: returnedSampleId, filepath, labelClass, labelSource, labelProbability } = await this.api.loadSample(sampleId);
            
            // Step 3: Delete annotation
            await this.api.deleteAnnotation(sampleId);
            
            // Step 4: Update UI with the loaded sample
            this.viewer.loadImage(imageUrl, filepath);
            const cls = null; // After deletion, there's no annotation
            await this.classManager.setCurrentSample(returnedSampleId, filepath, cls);
            if (this.updatePrediction) {
                this.updatePrediction(labelClass, labelProbability, labelSource);
            }
            
            // Step 5: Get stats
            if (this.updateStats) {
                await this.updateStats();
            }
            
            // Step 6: Get config (refresh from backend)
            await this.classManager.loadClassesFromConfig();
            
            // Step 7: Update training curve
            if (this.updateTrainingCurve) {
                await this.updateTrainingCurve();
            }
            
        } catch (e) {
            console.error('Undo workflow failed:', e);
        }
    }
}
