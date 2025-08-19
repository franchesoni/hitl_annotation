// statsView.js - Handles stats display and prediction updates

export class StatsView {
  constructor(api, classesView) {
    this.api = api;
    this.classesView = classesView;
    this.statsDiv = document.getElementById('stats-display');
    this.predictionDiv = document.getElementById('prediction-display');
    this.accuracySlider = document.getElementById('accuracy-slider');
    this.accuracyValue = document.getElementById('accuracy-slider-value');
    this.accuracyPct = this.accuracySlider ? Number(this.accuracySlider.value) : 100;
    if (this.accuracyValue) this.accuracyValue.textContent = `${this.accuracyPct}%`;
    this.statsRequestId = 0;

    if (this.accuracySlider) {
      this.accuracySlider.addEventListener('input', () => {
        this.accuracyPct = Number(this.accuracySlider.value);
        if (this.accuracyValue) this.accuracyValue.textContent = `${this.accuracyPct}%`;
        this.update();
      });
    }
  }


  updatePrediction(labelClass, labelProbability, labelSource) {
    if (!this.predictionDiv) return;
    if (labelSource === 'prediction' && labelClass) {
      const prob = labelProbability ? Number(labelProbability) : null;
      const pct = prob !== null && !isNaN(prob) ? (prob * 100).toFixed(1) : null;
      const text = pct !== null ? `${labelClass} (${pct}%)` : labelClass;
      this.predictionDiv.innerHTML = `<b>Prediction:</b> <span class="prediction-badge">${text}</span>`;
      if (this.classesView && typeof this.classesView.setPrediction === 'function') {
        this.classesView.setPrediction(labelClass);
      }
    } else {
      this.predictionDiv.innerHTML = '';
      if (this.classesView && typeof this.classesView.setPrediction === 'function') {
        this.classesView.setPrediction(null);
      }
    }
  }

  async update(currentImageFilename) {
    if (!this.statsDiv) return;
    const requestId = ++this.statsRequestId;
    try {
      const stats = await this.api.getStats(this.accuracyPct);
      if (requestId !== this.statsRequestId) return;
      if (stats) {
        let html = '';
        if (currentImageFilename) {
          html += `<div><b>Current image:</b> <span title="${currentImageFilename}">${currentImageFilename}</span></div>`;
        }
        if (stats.image) {
          html += `<div><b>Last image:</b> ${stats.image}</div>`;
        }
        html += `<div><b>Annotated:</b> ${stats.annotated}/${stats.total}</div>`;
        if (stats.annotation_counts) {
          const counts = Object.entries(stats.annotation_counts)
            .map(([cls, n]) => `<div>${cls}: ${n}</div>`)
            .join('');
          html += `<div><b>Annotations per class:</b>${counts}</div>`;
        }
        html += `<div><b>Tries:</b> ${stats.tries}</div>`;
        html += `<div><b>Correct:</b> ${stats.correct}</div>`;
        if (typeof stats.accuracy === 'number') {
          const pct = (stats.accuracy * 100).toFixed(1);
          html += `<div><b>Val Accuracy:</b> <span class="accuracy-badge">${pct}%</span></div>`;
        } else {
          html += `<div><b>Val Accuracy:</b> <span class="accuracy-badge">0%</span></div>`;
        }
        this.statsDiv.innerHTML = html;
      }
    } catch (e) {
      if (requestId === this.statsRequestId) {
        console.error('Failed to fetch stats:', e);
      }
    }
  }
}
