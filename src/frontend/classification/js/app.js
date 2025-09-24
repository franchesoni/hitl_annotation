import { API, buildNextParams } from '/shared/js/api.js';
import { ImageView } from '/shared/views/imageView.js';
import { ClassesView } from '/shared/views/classesView.js';
import { StatsView } from '/shared/views/statsView.js';
import { StrategyView } from '/shared/views/strategyView.js';
import { AIControlsView } from '/shared/views/aiControlsView.js';
import { TrainingCurveView } from '/shared/views/trainingCurveView.js';
import { SampleFilterView } from '/shared/views/sampleFilterView.js';
import { SampleInfoView } from '/shared/views/sampleInfoView.js';
import { Hotkeys } from '/shared/js/hotkeys.js';

document.addEventListener('DOMContentLoaded', async () => {
        // -----------------------------------------------------------
        // ----------  STATE  ----------------------------------------
        // -----------------------------------------------------------
        const state = {
                config: {
                        classes: [],
                        aiShouldBeRun: false,
                        architecture: 'resnet18',
                        budget: 1000,
                        resize: 224,
                        available_architectures: [],
                        samplePathFilter: '',
                        sampleFilterCount: null
                },
                history: [],
                configUpdated: false,
                workflowInProgress: false,
                currentImageFilepath: null,
                currentStats: null,  // Store current stats for optimization
                lastAnnotatedClass: null  // Track the last class annotated by the user
        };
        // -----------------------------------------------------------
        // ----------  COMPONENTS  -----------------------------------
        // -----------------------------------------------------------
        const leftPanel = document.querySelector('.left-panel');
        const classPanel = document.querySelector('#class-manager');
        const sampleFilterInput = document.getElementById('sample-filter-input');
        const sampleFilterCountEl = document.getElementById('sample-filter-count');
        const sampleInfoContainer = document.getElementById('sample-info');
        const api = new API();
        let aiControlsView = null;
        const sampleFilterView = new SampleFilterView({
                inputEl: sampleFilterInput,
                countEl: sampleFilterCountEl,
                api,
                state,
                onConfigApplied: () => {
                        if (aiControlsView) {
                                aiControlsView.render(state.config);
                        }
                }
        });

        function applyConfigFromServer(cfg) {
                if (!cfg) return;
                const sortedClasses = Array.isArray(cfg.classes) ? [...cfg.classes].sort() : [];
                state.config = {
                        ...state.config,
                        classes: sortedClasses,
                        aiShouldBeRun: !!cfg.ai_should_be_run,
                        architecture: typeof cfg.architecture === 'string' && cfg.architecture ? cfg.architecture : (state.config.architecture || 'resnet18'),
                        budget: typeof cfg.budget === 'number' ? cfg.budget : (state.config.budget ?? 1000),
                        resize: typeof cfg.resize === 'number' ? cfg.resize : (state.config.resize ?? 224),
                        available_architectures: Array.isArray(cfg.available_architectures) ? cfg.available_architectures : []
                };
                sampleFilterView.applyServerConfig(cfg);
                state.configUpdated = false;
        }

        await loadConfigFromServer();
        const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');
        const sampleInfoView = sampleInfoContainer ? new SampleInfoView({ container: sampleInfoContainer, state }) : null;
        const classesView = new ClassesView(classPanel, annotateWorkflow, state);
        const statsView = new StatsView(api, classesView);
        const trainingCurveView = new TrainingCurveView(api);
        const strategyView = new StrategyView();
        aiControlsView = new AIControlsView(api, state);
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
        async function annotateWorkflow(sampleId, className) {
                if (state.workflowInProgress) return; // guard
                beginWorkflow();
                try {
                        await updateConfigIfNeeded();
                        await api.annotateSample(sampleId, className);
                        state.history.push(sampleId);
                        state.lastAnnotatedClass = className;  // Track the last annotated class
                        await loadNextImage();
                        await getStatsAndConfig();
                } finally {
                        endWorkflow();
                }
        }

        // update strategyView list of classes when classes are updated
        classesView.setOnClassesUpdate((classes) => {
                strategyView.updateClasses(classes);
        });
        async function updateConfigIfNeeded() {
                if (state.configUpdated) {
                        await api.updateConfig(state.config);
                        state.configUpdated = false;  // config synchronized with backend
                }
        }
        const getStrategyParams = () => buildNextParams(strategyView, state.lastAnnotatedClass);
        
        // Removed minority_frontier_optimized hack to match API spec
        async function loadNextImage() {
                const { strategy, selectedClass } = getStrategyParams();
                const { imageUrl, sampleId, filepath, predictions } =
                        await api.loadNextImage(null, strategy, selectedClass);
                state.currentImageFilepath = filepath; // Store current image filepath in state
                if (sampleInfoView) {
                        sampleInfoView.update({ sampleId, filepath });
                }
                imageView.loadImage(imageUrl, filepath);
                await classesView.setCurrentSample(sampleId, filepath);
                statsView.updatePrediction(predictions);
        }
        async function loadConfigFromServer() {
                try {
                        // Silent bug fix: If there are local pending config changes (e.g., newly added classes)
                        // they'd previously be overwritten by the server response here. We now flush them first.
                        await updateConfigIfNeeded();
                        const cfg = await api.getConfig();
                        applyConfigFromServer(cfg);
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
                sampleFilterView.render();
        }
        const skipBtn = document.getElementById('skip-btn');
        async function skipCurrentSample() {
                if (state.workflowInProgress) return;
                const sampleId = classesView.currentSampleId;
                if (!sampleId) return;
                beginWorkflow();
                try {
                        await updateConfigIfNeeded();
                        await api.skipSample(sampleId);
                        state.history.push(sampleId);
                        await loadNextImage();
                        await getStatsAndConfig();
                } catch (e) {
                        console.error('Skip workflow failed:', e);
                } finally {
                        endWorkflow();
                }
        }
        if (skipBtn) {
                skipBtn.addEventListener('click', skipCurrentSample);
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
                        // 1) Push any pending config
                        await updateConfigIfNeeded();
                        // 2) Delete annotation for the current id
                        await api.deleteAnnotation(sampleId);
                        const prev = await api.loadSample(sampleId);
                        if (!prev) {
                                alert('Unable to reload undone sample');
                                return;
                        }
                        const { imageUrl, sampleId: returnedSampleId, filepath, predictions } = prev;
                        state.currentImageFilepath = filepath;
                        state.lastAnnotatedClass = null;
                        if (sampleInfoView) {
                                sampleInfoView.update({ sampleId: returnedSampleId, filepath });
                        }
                        imageView.loadImage(imageUrl, filepath);
                        await classesView.setCurrentSample(returnedSampleId, filepath);
                        statsView.updatePrediction(predictions);
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
                const hk = new Hotkeys();
                const guard = (fn) => (e) => { if (state.workflowInProgress) return; fn(e); };
                hk
                  .bind('ctrl+e', guard(() => api.exportDB()))
                  .bind('meta+e', guard(() => api.exportDB()))
                  .bind('ctrl+z', guard(() => undo()))
                  .bind('meta+z', guard(() => undo()))
                  .bind('backspace', guard(() => undo()))
                  .bind('u', guard(() => undo()))
                  .bind('s', guard(() => skipCurrentSample()))
                  .attach();
        }
        initKeyboard(api);
        try {
                await loadNextImage(); // now respects chosen strategy
                await getStatsAndConfig();
        } catch (e) {
                console.error('Failed to initialize application:', e);
        }
});
