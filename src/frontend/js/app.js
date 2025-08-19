import { ImageView } from './views/imageView.js';
import { ClassesView } from './views/classesView.js';
import { API } from './api.js';
import { StatsView } from './views/statsView.js';
import { StrategyView } from './views/strategyView.js';
import { AIControlsView } from './views/aiControlsView.js';
import { TrainingCurveView } from './views/trainingCurveView.js';

document.addEventListener('DOMContentLoaded', async () => {
        const leftPanel = document.querySelector('.left-panel');
        const classPanel = document.querySelector('#class-manager');
        const state = { 
                config: { classes: [], aiShouldBeRun: false, architecture: 'resnet18', budget: 1000, sleep: 0, resize: 224 }, 
                history: [], 
                configUpdated: false 
        };
        await loadConfigFromServer();  // Load initial config from server

        const api = new API();
        const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');
        
        // Create annotation workflow function
        async function annotateWorkflow(sampleId, className) {
                try {
                        await updateConfigIfNeeded();
                        await api.annotateSample(sampleId, className);
                        await loadNextImage();
                        await getStatsAndConfig();
                } catch (e) {
                        console.error('Annotation workflow failed:', e);
                        throw e;
                }
        }
        
        const classesView = new ClassesView(classPanel, annotateWorkflow, state);
        const statsView = new StatsView(api, classesView);
        const trainingCurveView = new TrainingCurveView(api);
        const strategyView = new StrategyView();
        const aiControlsView = new AIControlsView(api, state);

        // Set up ClassesView callbacks
        classesView.setOnClassChange((sampleId) => {
                if (sampleId) state.history.push(sampleId);
        });
        
        classesView.setOnClassesUpdate((classes) => {
                strategyView.updateClasses(classes);
        });

        // -------------------------------------------------------------------------------
        // CONTROLLER UTILS
        // -------------------------------------------------------------------------------
        // config update if needed
        async function updateConfigIfNeeded() {
                if (state.configUpdated) {
                        await api.updateConfig(state.config);
                        state.configUpdated = false;
                }
        }

        // Load next image workflow
        async function loadNextImage(strategy = null, pick = null) {
                const { imageUrl, sampleId, filepath, labelClass, labelSource, labelProbability } = 
                        await api.loadNextImage(null, strategy, pick);
                
                imageView.loadImage(imageUrl, filepath);
                await classesView.setCurrentSample(sampleId, filepath);
                statsView.updatePrediction(labelClass, labelProbability, labelSource);
        }


        // Load complete config from server
        async function loadConfigFromServer() {
                try {
                        const cfg = await api.getConfig();
                        if (cfg) {
                                state.config = {
                                        classes: cfg.classes || [],
                                        aiShouldBeRun: cfg.ai_should_be_run || false,
                                        architecture: cfg.architecture || 'resnet18',
                                        budget: cfg.budget || 1000,
                                        sleep: cfg.sleep || 0,
                                        resize: cfg.resize || 224,
                                };
                        }
                        state.configUpdated = false;
                } catch (e) {
                        console.error('Failed to load config from server:', e);
                }
        }



        // get stats, render, get config, render
        async function getStatsAndConfig() {
                await statsView.update();
                await trainingCurveView.update();
                await loadConfigFromServer();
                classesView.render();
                aiControlsView.render(state.config);
        }

        // -------------------------------------------------------------------------------
        // UNDO WORKFLOW
        // -------------------------------------------------------------------------------
        const undoBtn = document.getElementById('undo-btn');
        async function undo() {
                if (state.history.length === 0) {
                        alert('No more actions to undo');
                        return;
                }
                const sampleId = state.history.pop();
                try {
                        await updateConfigIfNeeded();
                        const { imageUrl, sampleId: returnedSampleId, filepath, labelClass, labelSource, labelProbability } = await api.loadSample(sampleId);
                        imageView.loadImage(imageUrl, filepath);
                        await classesView.setCurrentSample(returnedSampleId, filepath);
                        statsView.updatePrediction(labelClass, labelProbability, labelSource);
                        await api.deleteAnnotation(sampleId);
                        await getStatsAndConfig();
                } catch (e) {
                        console.error('Undo workflow failed:', e);
                }
        }
        if (undoBtn) {
                undoBtn.addEventListener('click', undo);
        }

        // -------------------------------------------------------------------------------
        // KEYBOARD SHORTCUTS
        // -------------------------------------------------------------------------------
        function initKeyboard(api) {
                document.addEventListener('keydown', (e) => {
                        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
                        const lowerCaseKey = e.key.toLowerCase();
                        if ((e.ctrlKey || e.metaKey) && lowerCaseKey === 'e') {
                                e.preventDefault();
                                api.exportDB();
                        } else if ((e.ctrlKey || e.metaKey) && lowerCaseKey === 'z') {
                                e.preventDefault();
                                undo();
                        } else if (e.key === 'Backspace' || lowerCaseKey === 'u') {
                                e.preventDefault();
                                undo();
                        }
                });
        }
        initKeyboard(api);

        // -------------------------------------------------------------------------------
        // START
        // -------------------------------------------------------------------------------       
        try {
                await loadNextImage();
                await getStatsAndConfig();
        } catch (e) {
                console.error('Failed to initialize application:', e);
        }
});
