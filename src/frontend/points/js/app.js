import { API } from '/shared/js/api.js';
import { ImageView } from '/shared/views/imageView.js';
import { ClassesView } from '/shared/views/classesView.js';
import { StatsView } from '/shared/views/statsView.js';
import { AIControlsView } from '/shared/views/aiControlsView.js';
import { TrainingCurveView } from '/shared/views/trainingCurveView.js';

document.addEventListener('DOMContentLoaded', async () => {
        // -----------------------------------------------------------
        // ----------  STATE  ----------------------------------------
        // -----------------------------------------------------------
        const state = {
                config: { classes: [], aiShouldBeRun: false, architecture: 'resnet18', budget: 1000, sleep: 0, resize: 224 },
                history: [],
                configUpdated: false,
                workflowInProgress: false,
                currentImageFilepath: null,
                currentStats: null,  // Store current stats for optimization
                selectedClass: null,
        };
        // -----------------------------------------------------------
        // ----------  COMPONENTS  -----------------------------------
        // -----------------------------------------------------------
        const leftPanel = document.querySelector('.left-panel');
        const classPanel = document.querySelector('#class-manager');
        const api = new API();
        await loadConfigFromServer();
        const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');
        const classesView = new ClassesView(classPanel, selectClassWorkflow, state);
        const statsView = new StatsView(api, classesView);
        const trainingCurveView = new TrainingCurveView(api);
        const aiControlsView = new AIControlsView(api, state);
        // -----------------------------------------------------------
        // ----------  ACTIONS  --------------------------------------
        // -----------------------------------------------------------

        function setInteractiveEnabled(enabled) {
                const elems = document.querySelectorAll('button, input, select, textarea');
                elems.forEach(el => {
                        if (!enabled) {
                                if (!el.disabled) el.dataset.prevDisabled = '1';
                                el.disabled = true;
                        } else if (el.dataset.prevDisabled === '1') {
                                el.disabled = false;
                                delete el.dataset.prevDisabled;
                        }
                });
        }
        function beginWorkflow() {
                state.workflowInProgress = true;
                setInteractiveEnabled(false);
        }
        function endWorkflow() {
                state.workflowInProgress = false;
                setInteractiveEnabled(true);
        }
        
        // Simple workflow that just selects a class without annotating
        async function selectClassWorkflow(sampleId, className) {
                // Just update the selected class, don't annotate yet
                state.selectedClass = className;
                console.log('Selected class:', className);
        }
        
        async function annotateWorkflow(sampleId, className) {
                if (state.workflowInProgress) return; // guard
                beginWorkflow();
                try {
                        await updateConfigIfNeeded();
                        await api.annotateSample(sampleId, className);
                        state.history.push(sampleId);
                        await loadNextImage();
                        await getStatsAndConfig();
                } finally {
                        endWorkflow();
                }
        }

        async function updateConfigIfNeeded() {
                if (state.configUpdated) {
                        await api.updateConfig(state.config);
                        state.configUpdated = false;  // config synchronized with backend
                }
        }

        async function loadNextImage() {
                const { imageUrl, sampleId, filepath, labelClass, labelSource, labelProbability } =
                        await api.loadNextImage();
                state.currentImageFilepath = filepath; // Store current image filepath in state
                imageView.loadImage(imageUrl, filepath);
                await classesView.setCurrentSample(sampleId, filepath);
                // Note: No prediction display for points annotation
        }
        async function loadConfigFromServer() {
                try {
                        // Silent bug fix: If there are local pending config changes (e.g., newly added classes)
                        // they'd previously be overwritten by the server response here. We now flush them first.
                        await updateConfigIfNeeded();
                        const cfg = await api.getConfig();
                        if (cfg) {
                                state.config = {
                                        classes: cfg.classes || [],
                                        aiShouldBeRun: cfg.ai_should_be_run || false,
                                        architecture: cfg.architecture || 'resnet18',
                                        budget: cfg.budget || 1000,
                                        sleep: cfg.sleep || 0,
                                        resize: cfg.resize || 224,
                                        available_architectures: cfg.available_architectures || []
                                };
                        }
                        state.configUpdated = false; // config synchronized with backend
                } catch (e) {
                        console.error('Failed to load config from server:', e);
                }
        }
        async function getStatsAndConfig() {
                const stats = await api.getStats();
                state.currentStats = stats;  // Store stats for optimization
                await statsView.update(stats, state.currentImageFilepath);
                await trainingCurveView.update(stats.training_stats);
                await loadConfigFromServer();
                classesView.render();
                aiControlsView.render(state.config);
        }
        const undoBtn = document.getElementById('undo-btn');
        async function undo() {
                if (state.workflowInProgress) return;
                if (state.history.length === 0) {
                        alert('No more actions to undo');
                        return;
                }
                const sampleId = state.history.pop();
                beginWorkflow();
                try {
                        await updateConfigIfNeeded();
                        const { imageUrl, sampleId: returnedSampleId, filepath, labelClass, labelSource, labelProbability } = await api.loadSample(sampleId);
                        state.currentImageFilepath = filepath; // Store current image filepath in state
                        imageView.loadImage(imageUrl, filepath);
                        await classesView.setCurrentSample(returnedSampleId, filepath);
                        // Note: No prediction display for points annotation
                        await api.deleteAnnotation(sampleId);
                        await getStatsAndConfig();
                } catch (e) {
                        console.error('Undo workflow failed:', e);
                } finally {
                        endWorkflow();
                }
        }
        if (undoBtn) {
                undoBtn.addEventListener('click', undo);
        }
        function initKeyboard(api) {
                document.addEventListener('keydown', (e) => {
                        if (state.workflowInProgress) return; // block all shortcuts during workflows
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
        try {
                await loadNextImage();
                await getStatsAndConfig();
        } catch (e) {
                console.error('Failed to initialize application:', e);
        }
});