import { ImageView } from './views/imageView.js';
import { ClassesView } from './views/classesView.js';
import { API } from './api.js';
import { AppController } from './controllers.js';
import { StatsView } from './views/statsView.js';
import { StrategyView } from './views/strategyView.js';
import { AIControlsView } from './views/aiControlsView.js';
import { TrainingCurveView } from './views/trainingCurveView.js';
import { initKeyboard } from './keyboard.js';

// Bootstraps everything: creates store, controllers, mounts views

document.addEventListener('DOMContentLoaded', async () => {
  const leftPanel = document.querySelector('.left-panel');
  const classPanel = document.querySelector('#class-manager');
  const undoBtn = document.getElementById('undo-btn');

  if (!leftPanel || !classPanel) {
    console.error('Required containers not found.');
    return;
  }

  const api = new API();
  const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');
  const statsView = new StatsView(api);
  const trainingCurveView = new TrainingCurveView(api);
  const strategyView = new StrategyView();
  const controller = new AppController(api, imageView, null, statsView, trainingCurveView, strategyView);
  const classesView = new ClassesView(classPanel, controller.annotateWorkflow.bind(controller), api);
  controller.setClassManager(classesView);
  statsView.setClassManager(classesView);

  const state = { aiRunning: false };
  const history = [];

  async function undo() {
    if (history.length === 0) {
      alert('No more actions to undo');
      return;
    }
    const sampleId = history.pop();
    try {
      await api.updateConfig({ classes: classesView.globalClasses });
      const { imageUrl, sampleId: returnedSampleId, filepath, labelClass, labelSource, labelProbability } = await api.loadSample(sampleId);
      await api.deleteAnnotation(sampleId);
      imageView.loadImage(imageUrl, filepath);
      const cls = null;
      await classesView.setCurrentSample(returnedSampleId, filepath, cls);
      statsView.updatePrediction(labelClass, labelProbability, labelSource);
      await statsView.update();
      await classesView.loadClassesFromConfig();
      await trainingCurveView.update();
    } catch (e) {
      console.error('Undo workflow failed:', e);
    }
  }

  if (undoBtn) {
    undoBtn.addEventListener('click', undo);
  }
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const key = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && key === 'z') {
      e.preventDefault();
      undo();
    } else if (e.key === 'Backspace' || key === 'u') {
      e.preventDefault();
      undo();
    }
  });

  classesView.setOnClassChange((sampleId) => {
    if (sampleId) history.push(sampleId);
  });
  classesView.setOnAnnotationSuccess((sampleId, cls) => {
    controller.setLastAnnotatedClass(cls);
  });
  classesView.setOnClassesUpdate((classes) => {
    strategyView.updateClasses(classes);
  });

  new AIControlsView(api, state);
  initKeyboard(api);

  try {
    await controller.loadNextImage();
  } catch (e) {
    console.error('Failed to fetch first image:', e);
  }
});
