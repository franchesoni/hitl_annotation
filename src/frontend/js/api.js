export class API {
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
    async annotateSample(sampleId, className) {
        const res = await fetch(`/api/annotate/${sampleId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ class: className })
        });
        if (!res.ok) throw new Error('Annotation failed');
        return await res.json();
    }
    async deleteAnnotation(sampleId) {
        const res = await fetch(`/api/annotate/${sampleId}`, {
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
        const labelClass = res.headers.get('X-Label-Class');
        const labelSource = res.headers.get('X-Label-Source');
        const labelProbability = res.headers.get('X-Label-Probability');
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        return { imageUrl, sampleId: sampleIdFromHeader, filepath, labelClass, labelSource, labelProbability };
    }
    async getConfig() {
        const res = await fetch('/api/config');
        if (!res.ok) throw new Error('Failed to get config');
        return await res.json();
    }
    async updateConfig(config) {
        const res = await fetch('/api/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error('Failed to update config');
        return await res.json();
    }
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
