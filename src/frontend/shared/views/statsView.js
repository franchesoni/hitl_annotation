export class StatsView {
  constructor(api, classesView) {
    this.api = api;
    this.classesView = classesView;
    this.statsDiv = document.getElementById('stats-display');
    this.predictionDiv = document.getElementById('prediction-display');
    this.statsRequestId = 0;
    this.currentStats = null; // cache of latest stats provided by the app
  }
  updatePrediction(predictionsOrLabelClass, labelProbability = null, labelSource = null) {
    if (!this.predictionDiv) return;
    let label = null;
    let prob01 = null;
    // New path: predictions object
    if (predictionsOrLabelClass && typeof predictionsOrLabelClass === 'object' && 'type' in predictionsOrLabelClass) {
      const preds = predictionsOrLabelClass;
      if (preds.type === 'label') {
        label = preds.label || null;
        if (preds.probability_ppm != null) {
          const v = Number(preds.probability_ppm);
          prob01 = isNaN(v) ? null : Math.max(0, Math.min(1, v / 1_000_000));
        }
      }
    } else {
      // Back-compat: old args (labelClass, labelProbability [0..1], labelSource)
      if (labelSource === 'prediction' && predictionsOrLabelClass) {
        label = predictionsOrLabelClass;
        prob01 = (typeof labelProbability === 'number') ? labelProbability : null;
      }
    }

    if (label) {
      const pctText = (typeof prob01 === 'number') ? ` (${(prob01 * 100).toFixed(1)}%)` : '';
      this.predictionDiv.innerHTML = `<b>Prediction:</b> <span class="prediction-badge">${label}${pctText}</span>`;
      if (this.classesView && typeof this.classesView.setPrediction === 'function') {
        this.classesView.setPrediction(label);
      }
    } else {
      this.predictionDiv.innerHTML = '';
      if (this.classesView && typeof this.classesView.setPrediction === 'function') {
        this.classesView.setPrediction(null);
      }
    }
  }
  async update(stats = null, currentImageFilename = null) {
    if (!this.statsDiv) return;
    const requestId = ++this.statsRequestId;

    // If stats are provided by caller, cache them. Otherwise reuse cached stats.
    if (stats) {
      this.currentStats = stats;
    } else {
      stats = this.currentStats;
    }
    
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
      this.statsDiv.innerHTML = html;
    }
  }
  
  // Removed live accuracy calculation for segmentation frontend
}
