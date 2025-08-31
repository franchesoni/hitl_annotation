import { API } from '/shared/js/api.js';
import { ImageView } from '/shared/views/imageView.js';
import { PointsClassesView } from './views/pointsClassesView.js';
import { StatsView } from '/shared/views/statsView.js';
import { StrategyView } from '/shared/views/strategyView.js';
import { AIControlsView } from '/shared/views/aiControlsView.js';
import { TrainingCurveView } from '/shared/views/trainingCurveView.js';

document.addEventListener('DOMContentLoaded', async () => {
    // -----------------------------------------------------------
    // ----------  STATE  ----------------------------------------
    // -----------------------------------------------------------
    const state = {
        config: { classes: [], aiShouldBeRun: false, architecture: 'small', budget: 1000, resize: 224 },
        configUpdated: false,
        workflowInProgress: false,
        currentImageFilepath: null,
        currentSampleId: null, // Track current sample ID for navigation
        currentStats: null,  // Store current stats for optimization
        selectedClass: null,
        classColors: new Map(), // Map class names to colors
        localPoints: [] // Local per-image list of points {class, x, y}
    };

    // -----------------------------------------------------------
    // ----------  COMPONENTS  -----------------------------------
    // -----------------------------------------------------------
    const leftPanel = document.querySelector('.left-panel');
    const classPanel = document.querySelector('#class-manager');
    const api = new API();
    await loadConfigFromServer();
    const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');

    // Overlay slider controls prediction/annotation overlay strength (0-100%)
    const overlaySlider = document.getElementById('overlay-slider');
    const overlayValue = document.getElementById('overlay-slider-value');
    if (overlaySlider) {
        const applyOverlay = () => {
            const pct = Number(overlaySlider.value || 0);
            if (overlayValue) overlayValue.textContent = `${pct}%`;
            // Store on imageView for future overlay rendering logic
            imageView.overlayAlpha = Math.max(0, Math.min(1, pct / 100));
        };
        overlaySlider.addEventListener('input', applyOverlay);
        applyOverlay();
    }

    // Set up point addition callback: local only
    imageView.onPointAdd = async (point) => {
        state.localPoints.push({ class: point.className, x: point.x, y: point.y });
        console.log('Point added locally:', point);
    };

    // Set up point removal callback: local only
    imageView.onPointRemove = async (point, index) => {
        console.log('Point removed locally:', point, 'at index:', index);
        if (index >= 0 && index < state.localPoints.length) {
            state.localPoints.splice(index, 1);
        } else {
            const TH = 1e-6;
            const i = state.localPoints.findIndex(p => p.class === point.className && Math.abs(p.x - point.x) < TH && Math.abs(p.y - point.y) < TH);
            if (i >= 0) state.localPoints.splice(i, 1);
        }
    };
    const classesView = new PointsClassesView(classPanel, selectClassWorkflow, state);
    const statsView = new StatsView(api, classesView);
    const trainingCurveView = new TrainingCurveView(api);
    // Use segmentation-specific architectures for points annotation
    const segmentationArchitectures = ['small', 'small+', 'base', 'large'];
    const aiControlsView = new AIControlsView(api, state, segmentationArchitectures);
    const strategyView = new StrategyView();

    // Utility to retrieve the color for a class from the view so that
    // all components use the same color palette and assignments.
    function getClassColor(className) {
        return classesView.getClassColor(className);
    }

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
        // Ask the view for the color so it stays the single source of truth
        const color = classesView.getClassColor ? classesView.getClassColor(className) : undefined;
        console.log('Selected class:', className, 'with color:', color);

        // Update the image view with the selected class and color
        imageView.setSelectedClass(className, color);

        // You can also trigger visual updates here if needed
        // For example, update cursor style or selected point color preview
    }
    // Keep strategyView class list in sync with classes view
    if (classesView && typeof classesView.setOnClassesUpdate === 'function') {
        classesView.setOnClassesUpdate((classes) => strategyView.updateClasses(classes));
    }

    async function annotateWorkflow(sampleId, className) {
        if (state.workflowInProgress) return; // guard
        beginWorkflow();
        try {
            await updateConfigIfNeeded();
            await api.annotateSample(sampleId, className);

            await loadNextImage();
            await getStatsAndConfig();
            updateNavigationButtons();
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

    function getStrategyParams() {
        const uiStrategy = strategyView?.currentStrategy || null;
        if (uiStrategy === 'specific_class') {
            return { strategy: 'specific_class', selectedClass: strategyView.currentSpecificClass || null };
        }
        if (uiStrategy === 'last_class') {
            // For segmentation, use currently selected class as the last-class proxy
            return { strategy: 'specific_class', selectedClass: state.selectedClass || null };
        }
        return { strategy: uiStrategy, selectedClass: null };
    }

    async function loadNextImage() {
        const { strategy, selectedClass } = getStrategyParams();
        const { imageUrl, sampleId, filepath, predictions } = await api.loadNextImage(null, strategy, selectedClass);
        state.currentImageFilepath = filepath;
        state.currentSampleId = parseInt(sampleId);

        imageView.loadImage(imageUrl, filepath);
        imageView.clearPoints(); // Clear existing points when loading new image
        state.localPoints = [];
        state.localPoints = [];

        // Load existing point annotations from backend
        try {
            const annotationsData = await api.getAnnotations(sampleId);
            const pointAnnotations = annotationsData.annotations.filter(ann => ann.type === 'point');

            // Add existing points to the image view & local store
            pointAnnotations.forEach(ann => {
                const color = getClassColor(ann.class);
                const x = (typeof ann.col01 === 'number') ? ann.col01 / 1_000_000 : (ann.col ?? 0);
                const y = (typeof ann.row01 === 'number') ? ann.row01 / 1_000_000 : (ann.row ?? 0);
                imageView.addExistingPoint(x, y, ann.class, color);
                state.localPoints.push({ class: ann.class, x, y });
            });

            console.log(`Loaded ${pointAnnotations.length} existing points`);
        } catch (error) {
            console.error('Failed to load existing annotations:', error);
        }

        // If mask predictions are advertised, fetch mask assets
        if (predictions && predictions.type === 'mask' && predictions.mask_map) {
            await loadMaskAssets(predictions.mask_map);
        } else if (imageView) {
            imageView.maskOverlays = null;
        }

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
                    classes: (cfg.classes || []).sort(),
                    aiShouldBeRun: cfg.ai_should_be_run || false,
                    architecture: cfg.architecture || 'small', // Default to 'small' for segmentation
                    budget: cfg.budget || 1000,
                    resize: cfg.resize || 224,
                    available_architectures: cfg.available_architectures || []
                };
                if (state.classColors instanceof Map) {
                    state.classColors.clear();
                }
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

        // Update class colors when classes change
        state.config.classes.forEach(className => {
            getClassColor(className); // Ensures color is assigned
        });

        classesView.render();
        aiControlsView.render(state.config);
    }
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    async function savePointsForCurrentImage() {
        if (!state.currentSampleId) return;
        try {
            if (!state.localPoints || state.localPoints.length === 0) {
                // Explicitly clear points for this image when the set is empty
                await api.clearPoints(state.currentSampleId);
            } else {
                await api.savePointAnnotations(state.currentSampleId, state.localPoints);
            }
        } catch (e) {
            console.error('Failed to persist point annotations:', e);
        }
    }

    async function navigatePrev() {
        if (state.workflowInProgress) return;
        if (!state.currentSampleId) {
            alert('No current sample to navigate from');
            return;
        }

        beginWorkflow();
        try {
            await updateConfigIfNeeded();
            await savePointsForCurrentImage();
            const prevSample = await api.loadSamplePrev(state.currentSampleId);
            if (prevSample) {
                await loadSampleData(prevSample);
                await getStatsAndConfig();
            } else {
                alert('No previous image available');
            }
            updateNavigationButtons();
        } catch (e) {
            console.error('Navigate prev failed:', e);
        } finally {
            endWorkflow();
        }
    }

    async function navigateNext() {
        if (state.workflowInProgress) return;

        beginWorkflow();
        try {
            await updateConfigIfNeeded();
            await savePointsForCurrentImage();
            let nextSample = null;

            // Try to get next sample if we have a current sample
            if (state.currentSampleId) {
                nextSample = await api.loadSampleNext(state.currentSampleId);
            }

            // If no next sample found, load a new one using strategy params
            if (!nextSample) {
                await loadNextImage();
            } else {
                await loadSampleData(nextSample);
            }

            await getStatsAndConfig();
            updateNavigationButtons();
        } catch (e) {
            console.error('Navigate next failed:', e);
        } finally {
            endWorkflow();
        }
    }

    async function loadSampleData(sampleData) {
        const { imageUrl, sampleId, filepath, predictions } = sampleData;
        state.currentImageFilepath = filepath;
        state.currentSampleId = parseInt(sampleId);
        imageView.loadImage(imageUrl, filepath);
        imageView.clearPoints(); // Clear existing points when loading new image

        // Load existing point annotations from backend
        try {
            const annotationsData = await api.getAnnotations(sampleId);
            const pointAnnotations = annotationsData.annotations.filter(ann => ann.type === 'point');

            // Add existing points to the image view & local store
            pointAnnotations.forEach(ann => {
                const color = getClassColor(ann.class);
                const x = (typeof ann.col01 === 'number') ? ann.col01 / 1_000_000 : (ann.col ?? 0);
                const y = (typeof ann.row01 === 'number') ? ann.row01 / 1_000_000 : (ann.row ?? 0);
                imageView.addExistingPoint(x, y, ann.class, color);
                state.localPoints.push({ class: ann.class, x, y });
            });

            console.log(`Loaded ${pointAnnotations.length} existing points`);
        } catch (error) {
            console.error('Failed to load existing annotations:', error);
        }

        // If mask predictions are advertised, fetch mask assets
        if (predictions && predictions.type === 'mask' && predictions.mask_map) {
            await loadMaskAssets(predictions.mask_map);
        } else if (imageView) {
            imageView.maskOverlays = null;
        }

        await classesView.setCurrentSample(sampleId, filepath);
    }

    async function loadMaskAssets(maskMap) {
        const overlays = {};
        const overlayColors = {};
        const entries = Object.entries(maskMap || {});
        await Promise.all(entries.map(([cls, url]) => new Promise((resolve) => {
            const img = new Image();
            img.onload = () => { overlays[cls] = img; resolve(); };
            img.onerror = () => { console.warn('Failed to load mask asset', url); resolve(); };
            img.src = url;
        })));
        // Assign class colors for tinting
        for (const cls of Object.keys(overlays)) {
            overlayColors[cls] = getClassColor(cls);
        }
        imageView.maskOverlays = overlays;
        imageView.maskOverlayColors = overlayColors;
    }

    function updateNavigationButtons() {
        // Note: We can't easily determine if prev/next exist without making API calls
        // So we'll enable them and let the API calls handle the "not found" cases
        if (prevBtn) {
            prevBtn.disabled = !state.currentSampleId || state.workflowInProgress;
        }
        if (nextBtn) {
            nextBtn.disabled = state.workflowInProgress;
            nextBtn.textContent = 'Next ▶ (Ctrl+Right/N)';
        }
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', navigatePrev);
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', navigateNext);
    }
    function initKeyboard(api) {
        document.addEventListener('keydown', (e) => {
            if (state.workflowInProgress) return; // block all shortcuts during workflows
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const lowerCaseKey = e.key.toLowerCase();
            if ((e.ctrlKey || e.metaKey) && lowerCaseKey === 'e') {
                e.preventDefault();
                api.exportDB();
            } else if ((e.ctrlKey || e.metaKey) && e.key === 'ArrowLeft') {
                e.preventDefault();
                navigatePrev();
            } else if ((e.ctrlKey || e.metaKey) && e.key === 'ArrowRight') {
                e.preventDefault();
                navigateNext();
            } else if (lowerCaseKey === 'p') {
                e.preventDefault();
                navigatePrev();
            } else if (lowerCaseKey === 'n') {
                e.preventDefault();
                navigateNext();
            } else if (lowerCaseKey === 'u' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                // Undo last point
                const lastPoint = imageView.getLastPoint();
                if (lastPoint) {
                    imageView.removeLastPoint();
                    // Remove from local list too
                    state.localPoints.pop();
                }
            } else if (lowerCaseKey === 'c' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                // Clear all points
                imageView.clearPoints();
                state.localPoints = [];
            } else if (e.key >= '1' && e.key <= '9') {
                e.preventDefault();
                // Select class by number (1-9)
                const classIndex = parseInt(e.key) - 1;
                if (classIndex < state.config.classes.length) {
                    const className = state.config.classes[classIndex];
                    selectClassWorkflow(state.currentSampleId, className);
                }
            } else if (e.key === '0') {
                e.preventDefault();
                // Select 10th class (index 9)
                if (state.config.classes.length > 9) {
                    const className = state.config.classes[9];
                    selectClassWorkflow(state.currentSampleId, className);
                }
            }
        });
    }
    initKeyboard(api);
    try {
        await loadNextImage();
        await getStatsAndConfig();
        updateNavigationButtons(); // Initialize button states
    } catch (e) {
        console.error('Failed to initialize application:', e);
    }
});
