// controllers.js - Orchestrates interactions between views, store, and API

export class AppController {
  constructor(api, imageView, classManager, statsView, trainingCurveView, strategyView) {
    this.api = api;
    this.imageView = imageView;
    this.classManager = classManager;
    this.statsView = statsView;
    this.trainingCurveView = trainingCurveView;
    this.strategyView = strategyView;
    this.lastAnnotatedClass = null;
  }

  setClassManager(cm) {
    this.classManager = cm;
  }

  setLastAnnotatedClass(cls) {
    this.lastAnnotatedClass = cls;
  }

  async loadNextImage() {
    if (!this.classManager) return;
    const currentId = this.classManager.currentSampleId;
    let strategy = this.strategyView.currentStrategy;
    let pick = null;
    if (strategy === 'pick_class') {
      pick = this.lastAnnotatedClass;
      if (!pick) {
        strategy = 'sequential';
      }
    } else if (strategy === 'specific_class') {
      strategy = 'pick_class';
      pick = this.strategyView.currentSpecificClass;
    }
    try {
      const { imageUrl, sampleId, filepath, labelClass, labelSource, labelProbability } = await this.api.loadNextImage(currentId, strategy, pick);
      this.imageView.loadImage(imageUrl, filepath);
      const annClass = labelSource === 'annotation' ? labelClass : null;
      await this.classManager.setCurrentSample(sampleId, filepath, annClass);
      this.statsView.updatePrediction(labelClass, labelProbability, labelSource);
      await this.statsView.update();
      await this.trainingCurveView.update();
    } catch (e) {
      console.error('Failed to fetch next image:', e);
    }
  }

  async annotateWorkflow(sampleId, className) {
    try {
      await this.api.updateConfig({ classes: this.classManager.globalClasses });
      await this.api.annotateSample(sampleId, className);
      await this.loadNextImage();
      await this.statsView.update();
      await this.classManager.loadClassesFromConfig();
      await this.trainingCurveView.update();
    } catch (e) {
      console.error('Annotation workflow failed:', e);
      throw e;
    }
  }
}
