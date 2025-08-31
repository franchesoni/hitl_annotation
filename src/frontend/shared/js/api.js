export class API {
    _parseImageResponse = async (res) => {
        if (!res.ok) throw new Error('Image request failed');
        const sampleId = res.headers.get('X-Image-Id');
        const filepath = res.headers.get('X-Image-Filepath');
        const predictions = this._parsePredictionHeaders(res);
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        // Back-compat surface for classification view until it is updated
        let labelClass = null, labelProbability = null, labelSource = null;
        if (predictions && predictions.type === 'label') {
            labelClass = predictions.label;
            labelProbability = (predictions.probability_ppm != null)
                ? (Number(predictions.probability_ppm) / 1_000_000)
                : null;
            labelSource = 'prediction';
        }
        return { imageUrl, sampleId, filepath, predictions, labelClass, labelSource, labelProbability };
    }
    _parsePredictionHeaders(res) {
        const typeHeader = res.headers.get('X-Predictions-Type');
        const maskHeader = res.headers.get('X-Predictions-Mask');
        // Prefer mask header presence regardless of type header to avoid coupling
        if (maskHeader) {
            const maskJson = maskHeader;
            let parsed = null;
            try { parsed = maskJson ? JSON.parse(maskJson) : null; } catch (_) { parsed = null; }
            // Normalize to a plain object map: { [className]: url }
            let mask_map = null;
            if (parsed && Array.isArray(parsed)) {
                mask_map = {};
                for (const item of parsed) {
                    const cls = item && item.class;
                    const url = item && item.url;
                    if (typeof cls === 'string' && typeof url === 'string') mask_map[cls] = url;
                }
                if (Object.keys(mask_map).length === 0) mask_map = null;
            } else if (parsed && typeof parsed === 'object') {
                mask_map = {};
                for (const [k, v] of Object.entries(parsed)) { if (typeof v === 'string') mask_map[k] = v; }
                if (Object.keys(mask_map).length === 0) mask_map = null;
            }
            return { type: 'mask', mask_map };
        }
        const type = typeHeader || null;
        if (!type) return null;
        if (type === 'label') {
            const label = res.headers.get('X-Predictions-Label');
            const probPpm = res.headers.get('X-Predictions-Probability');
            const probability_ppm = probPpm !== null ? Number(probPpm) : null;
            return { type: 'label', label, probability_ppm };
        }
        // If type explicitly says mask but header missing, nothing to do
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
        return await this._parseImageResponse(res);
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
            class: p.className,
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
        // API spec: DELETE /api/annotations/<sampleId> deletes all annotations for the sample
        const res = await fetch(`/api/annotations/${sampleId}`, {
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
        return await this._parseImageResponse(res);
    }
    
    async loadSamplePrev(sampleId) {
        const res = await fetch(`/api/samples/${sampleId}/prev`);
        if (!res.ok) return null; // No previous sample
        return await this._parseImageResponse(res);
    }
    
    async loadSampleNext(sampleId) {
        const res = await fetch(`/api/samples/${sampleId}/next`);
        if (!res.ok) return null; // No next sample
        return await this._parseImageResponse(res);
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

// Build strategy params for next-image calls from UI state
// strategyViewState: { currentStrategy: string|null, currentSpecificClass?: string|null }
// lastClass: last class to use when strategy is 'last_class' (app-specific)
export function buildNextParams(strategyViewState, lastClass = null) {
    const uiStrategy = strategyViewState?.currentStrategy || null;
    if (uiStrategy === 'specific_class') {
        const pick = strategyViewState?.currentSpecificClass || null;
        return pick ? { strategy: 'specific_class', selectedClass: pick }
                    : { strategy: 'sequential', selectedClass: null };
    }
    if (uiStrategy === 'last_class') {
        const last = lastClass || null;
        return last ? { strategy: 'specific_class', selectedClass: last }
                    : { strategy: 'sequential', selectedClass: null };
    }
    return { strategy: uiStrategy, selectedClass: null };
}

// Optional convenience: alias for saving points with PPM conversion
export function savePoints(api, sampleId, points) {
    return api.savePointAnnotations(sampleId, points);
}
