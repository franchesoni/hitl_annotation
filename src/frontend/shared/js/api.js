export class API {
    _parsePredictionHeaders(res) {
        const type = res.headers.get('X-Predictions-Type');
        if (!type) return null;
        if (type === 'label') {
            const label = res.headers.get('X-Predictions-Label');
            const probPpm = res.headers.get('X-Predictions-Probability');
            const probability_ppm = probPpm !== null ? Number(probPpm) : null;
            return { type: 'label', label, probability_ppm };
        }
        if (type === 'mask') {
            const maskJson = res.headers.get('X-Predictions-Mask');
            let mask_map = null;
            try {
                mask_map = maskJson ? JSON.parse(maskJson) : null;
            } catch (_) {
                mask_map = null;
            }
            return { type: 'mask', mask_map };
        }
        return null;
    }
    async loadNextImage(currentId = null, strategy = null, pick = null) {
        let url = '/api/samples/next';
        const params = new URLSearchParams();
        if (strategy) {
            params.append('strategy', strategy);
        }
        if (pick) {
            // API spec expects `class` param when strategy=specific_class
            params.append('class', pick);
        }
        const qs = params.toString();
        if (qs) url += `?${qs}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('No images available');
        const sampleId = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const predictions = this._parsePredictionHeaders(res);
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        // Back-compat surface for classification view until it is updated
        let labelClass = null, labelProbability = null, labelSource = null;
        if (predictions && predictions.type === 'label') {
            labelClass = predictions.label;
            labelProbability = (predictions.probability_ppm != null) ? (Number(predictions.probability_ppm) / 1_000_000) : null;
            labelSource = 'prediction';
        }
        return { imageUrl, sampleId, filepath, predictions, labelClass, labelSource, labelProbability };
    }
    async annotateSample(sampleId, className) {
        const payload = [
            { type: 'label', class: className, timestamp: new Date().toISOString() }
        ];
        const res = await fetch(`/api/annotations/${sampleId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Annotation failed');
        return await res.json();
    }
    async savePointAnnotations(sampleId, points) {
        // points: [{ class: string, x: number, y: number }] with normalized coords [0,1]
        const payload = points.map(p => ({
            type: 'point',
            class: p.class,
            col01: Math.max(0, Math.min(1_000_000, Math.round((p.x ?? 0) * 1_000_000))),
            row01: Math.max(0, Math.min(1_000_000, Math.round((p.y ?? 0) * 1_000_000)))
        }));
        const res = await fetch(`/api/annotations/${sampleId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Point annotations save failed');
        return await res.json();
    }

    async clearPoints(sampleId) {
        const res = await fetch(`/api/annotations/${sampleId}?type=point`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error('Clear points failed');
        return await res.json();
    }
    
    async getAnnotations(sampleId) {
        const res = await fetch(`/api/annotations/${sampleId}`);
        if (!res.ok) throw new Error('Failed to get annotations');
        return await res.json();
    }
    async deleteAnnotation(sampleId) {
        const res = await fetch(`/api/annotations/${sampleId}`, {
            method: 'DELETE'
        });
        if (!res.ok) throw new Error('Delete failed');
        return await res.json();
    }
    async loadSample(sampleId) {
        const res = await fetch(`/api/samples/${sampleId}`);
        if (!res.ok) throw new Error('Sample not found');
        const sampleIdFromHeader = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const predictions = this._parsePredictionHeaders(res);
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        let labelClass = null, labelProbability = null, labelSource = null;
        if (predictions && predictions.type === 'label') {
            labelClass = predictions.label;
            labelProbability = (predictions.probability_ppm != null) ? (Number(predictions.probability_ppm) / 1_000_000) : null;
            labelSource = 'prediction';
        }
        return { imageUrl, sampleId: sampleIdFromHeader, filepath, predictions, labelClass, labelSource, labelProbability };
    }
    
    async loadSamplePrev(sampleId) {
        const res = await fetch(`/api/samples/${sampleId}/prev`);
        if (!res.ok) return null; // No previous sample
        const sampleIdFromHeader = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const predictions = this._parsePredictionHeaders(res);
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        let labelClass = null, labelProbability = null, labelSource = null;
        if (predictions && predictions.type === 'label') {
            labelClass = predictions.label;
            labelProbability = (predictions.probability_ppm != null) ? (Number(predictions.probability_ppm) / 1_000_000) : null;
            labelSource = 'prediction';
        }
        return { imageUrl, sampleId: sampleIdFromHeader, filepath, predictions, labelClass, labelSource, labelProbability };
    }
    
    async loadSampleNext(sampleId) {
        const res = await fetch(`/api/samples/${sampleId}/next`);
        if (!res.ok) return null; // No next sample
        const sampleIdFromHeader = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const predictions = this._parsePredictionHeaders(res);
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        let labelClass = null, labelProbability = null, labelSource = null;
        if (predictions && predictions.type === 'label') {
            labelClass = predictions.label;
            labelProbability = (predictions.probability_ppm != null) ? (Number(predictions.probability_ppm) / 1_000_000) : null;
            labelSource = 'prediction';
        }
        return { imageUrl, sampleId: sampleIdFromHeader, filepath, predictions, labelClass, labelSource, labelProbability };
    }
    async getConfig() {
        const res = await fetch('/api/config');
        if (!res.ok) throw new Error('Failed to get config');
        return await res.json();
    }

    async getArchitectures() {
        const cfg = await this.getConfig();
        return Array.isArray(cfg.available_architectures) ? cfg.available_architectures : [];
    }
    async updateConfig(config) {
        // Map frontend camelCase flag to backend snake_case if present
        const payload = { ...config };
        if (Object.prototype.hasOwnProperty.call(payload, 'aiShouldBeRun')) {
            payload.ai_should_be_run = payload.aiShouldBeRun;
            delete payload.aiShouldBeRun;
        }
        const res = await fetch('/api/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to update config');
        return await res.json();
    }

    async getStats() {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error('Failed to get stats');
        return await res.json();
    }

    async exportDB() {
        const res = await fetch('/api/export');
        if (!res.ok) throw new Error('Failed to export DB');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'annotations.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }
}
