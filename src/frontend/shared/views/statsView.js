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
        this.update(); // Call without parameters to fetch fresh stats with new percentage
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
  async update(stats = null, currentImageFilename = null) {
    if (!this.statsDiv) return;
    const requestId = ++this.statsRequestId;
    
    // If stats not provided, fetch them
    if (!stats) {
      stats = await this.api.getStats();
    }
    
    if (requestId !== this.statsRequestId) return;
    if (stats) {
      // Calculate windowed live accuracy stats
      const liveAccuracyStats = this.calculateLiveAccuracyStats(stats.live_accuracy_points || [], this.accuracyPct);
      
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
      html += `<div><b>Tries:</b> ${liveAccuracyStats.tries}</div>`;
      html += `<div><b>Correct:</b> ${liveAccuracyStats.correct}</div>`;
      if (typeof liveAccuracyStats.accuracy === 'number') {
        const pct = (liveAccuracyStats.accuracy * 100).toFixed(1);
        html += `<div><b>Live Accuracy:</b> <span class="accuracy-badge">${pct}%</span></div>`;
      } else {
        html += `<div><b>Live Accuracy:</b> <span class="accuracy-badge">0%</span></div>`;
      }
      this.statsDiv.innerHTML = html;
    }
  }
  
  calculateLiveAccuracyStats(liveAccuracyPoints, windowPercentage) {
    if (!liveAccuracyPoints || liveAccuracyPoints.length === 0) {
      return { tries: 0, correct: 0, accuracy: 0.0 };
    }
    
    const totalPoints = liveAccuracyPoints.length;
    const windowSize = Math.max(1, Math.floor(totalPoints * windowPercentage / 100));
    
    // Get the last windowSize points
    const windowPoints = liveAccuracyPoints.slice(-windowSize);
    
    const tries = windowPoints.length;
    const correct = windowPoints.filter(p => p.value === 1.0).length;
    const accuracy = tries > 0 ? correct / tries : 0.0;
    
    return { tries, correct, accuracy };
  }
}
