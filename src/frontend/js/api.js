// api.js - Centralized frontend API for backend communication

export class API {
    // Load the next image sample (image response, not JSON)
    async loadNextImage(currentId = null, strategy = null, pick = null) {
        let url = '/api/samples/next';
        const params = new URLSearchParams();
        if (strategy) {
            params.append('strategy', strategy);
        }
        if (pick) {
            params.append('pick', pick);
        }
        const qs = params.toString();
        if (qs) url += `?${qs}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('No images available');
        const sampleId = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const labelClass = res.headers.get('X-Label-Class');
        const labelSource = res.headers.get('X-Label-Source');
        const labelProbability = res.headers.get('X-Label-Probability');
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        return { imageUrl, sampleId, filepath, labelClass, labelSource, labelProbability };
    }

    // Annotate a sample (returns JSON status)
    async annotateSample(sampleId, className) {
        const res = await fetch(`/api/annotate/${sampleId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ class: className })
        });
        if (!res.ok) throw new Error('Annotation failed');
        return await res.json();
    }

    // Delete annotation for a sample
    async deleteAnnotation(sampleId) {
        const res = await fetch(`/api/annotate/${sampleId}`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error('Delete failed');
        return await res.json();
    }

    // Load specific sample by id
    async loadSample(sampleId) {
        const res = await fetch(`/api/samples/${sampleId}`);
        if (!res.ok) throw new Error('Sample not found');
        const sampleIdFromHeader = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const labelClass = res.headers.get('X-Label-Class');
        const labelSource = res.headers.get('X-Label-Source');
        const labelProbability = res.headers.get('X-Label-Probability');
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        return { imageUrl, sampleId: sampleIdFromHeader, filepath, labelClass, labelSource, labelProbability };
    }

    // Get config
    async getConfig() {
        const res = await fetch('/api/config');
        if (!res.ok) throw new Error('Failed to get config');
        return await res.json();
    }

    // Update config
    async updateConfig(config) {
        const res = await fetch('/api/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error('Failed to update config');
        return await res.json();
    }

    // Get accuracy stats with optional window percentage
    async getStats(pct = 100) {
        let url = '/api/stats';
        if (typeof pct === 'number' && pct !== 100) {
            const params = new URLSearchParams({ pct: pct.toString() });
            url += `?${params.toString()}`;
        }
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to get stats');
        return await res.json();
    }

    async getTrainingStats() {
        // Get training stats from the main stats endpoint
        const stats = await this.getStats();
        return stats.training_stats || [];
    }

    async exportDB() {
        const res = await fetch('/api/export');
        if (!res.ok) throw new Error('Failed to export DB');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'db_export.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }
}



//   annotateWorkflow(sampleId, className) {
//     try {
//       await this.api.updateConfig({ classes: this.classManager.globalClasses });
//       await this.api.annotateSample(sampleId, className);
//       await this.loadNextImage();
//       await this.statsView.update();
//       await this.classManager.loadClassesFromConfig();
//       await this.trainingCurveView.update();
//     } catch (e) {
//       console.error('Annotation workflow failed:', e);
//       throw e;
//     }
//   }

