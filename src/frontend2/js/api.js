// api.js - Centralized frontend API for backend communication

export class API {
    // Load the next image sample (image response, not JSON)
    async loadNextImage(currentId = null, strategy = null, className = null) {
        let url = '/next';
        const params = new URLSearchParams();
        if (currentId) {
            params.append('current_id', currentId);
        }
        if (strategy) {
            params.append('strategy', strategy);
        }
        if (className) {
            params.append('class', className);
        }
        const qs = params.toString();
        if (qs) url += `?${qs}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('No images available');
        const filename = res.headers.get('X-Image-Id') || res.headers.get('X-Filename');
        const labelClass = res.headers.get('X-Label-Class');
        const labelSource = res.headers.get('X-Label-Source');
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        return { imageUrl, filename, labelClass, labelSource };
    }

    // Annotate a sample (returns JSON status)
    async annotateSample(filepath, className) {
        const res = await fetch('/annotate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath, class: className })
        });
        if (!res.ok) throw new Error('Annotation failed');
        return await res.json();
    }

    // Delete annotation for a sample
    async deleteAnnotation(filepath) {
        const res = await fetch('/annotate', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath })
        });
        if (!res.ok) throw new Error('Delete failed');
        return await res.json();
    }

    // Load specific sample by id
    async loadSample(id) {
        const res = await fetch(`/sample?id=${encodeURIComponent(id)}`);
        if (!res.ok) throw new Error('Sample not found');
        const filename = res.headers.get('X-Image-Id') || res.headers.get('X-Filename');
        const labelClass = res.headers.get('X-Label-Class');
        const labelSource = res.headers.get('X-Label-Source');
        const blob = await res.blob();
        const imageUrl = URL.createObjectURL(blob);
        return { imageUrl, filename, labelClass, labelSource };
    }

    // Get config
    async getConfig() {
        const res = await fetch('/config');
        if (!res.ok) throw new Error('Failed to get config');
        return await res.json();
    }

    // Update config
    async updateConfig(config) {
        const res = await fetch('/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error('Failed to update config');
        return await res.json();
    }

    // Get accuracy stats with optional window percentage
    async getStats(pct = 100) {
        let url = '/stats';
        if (typeof pct === 'number') {
            const params = new URLSearchParams({ pct: pct.toString() });
            url += `?${params.toString()}`;
        }
        const res = await fetch(url);
        if (!res.ok) throw new Error('Failed to get stats');
        return await res.json();
    }

    async getTrainingStats() {
        const res = await fetch('/training_stats');
        if (!res.ok) throw new Error('Failed to get training stats');
        return await res.json();
    }

    async exportDB() {
        const res = await fetch('/export_db');
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
