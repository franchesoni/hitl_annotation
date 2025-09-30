import { API } from '/shared/js/api.js';
import { ImageView } from '/shared/views/imageView.js';
import { PointsClassesView } from './views/pointsClassesView.js';
import { StatsView } from '/shared/views/statsView.js';
import { AIControlsView } from '/shared/views/aiControlsView.js';
import { TrainingCurveView } from '/shared/views/trainingCurveView.js';
import { Hotkeys } from '/shared/js/hotkeys.js';
import { SampleFilterView } from '/shared/views/sampleFilterView.js';
import { SampleInfoView } from '/shared/views/sampleInfoView.js';

document.addEventListener('DOMContentLoaded', async () => {
    // -----------------------------------------------------------
    // ----------  STATE  ----------------------------------------
    // -----------------------------------------------------------
    const state = {
        config: {
            classes: [],
            aiShouldBeRun: false,
            architecture: 'small',
            budget: 1000,
            resize: 224,
            available_architectures: [],
            samplePathFilter: '',
            sampleFilterCount: null
        },
        configUpdated: false,
        workflowInProgress: false,
        currentImageFilepath: null,
        currentSampleId: null, // Track current sample ID for navigation
        currentStats: null,  // Store current stats for optimization
        selectedClass: null,
        classColors: new Map(), // Map class names to colors
        currentMaskPredictions: null,
        currentMaskAnnotations: null,
        isAcceptingMask: false,
        isRemovingMask: false,
    };

    // -----------------------------------------------------------
    // ----------  COMPONENTS  -----------------------------------
    // -----------------------------------------------------------
    const leftPanel = document.querySelector('.left-panel');
    const classPanel = document.querySelector('#class-manager');
    const api = new API();
    const sampleFilterInput = document.getElementById('sample-filter-input');
    const sampleFilterCountEl = document.getElementById('sample-filter-count');
    const sampleInfoContainer = document.getElementById('sample-info');
    const maskAnnotationToggle = document.getElementById('mask-annotation-toggle');
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
        state.config = {
            ...state.config,
            classes: (cfg.classes || []).sort(),
            aiShouldBeRun: cfg.ai_should_be_run || false,
            architecture: cfg.architecture || 'small', // Default to 'small' for segmentation
            budget: cfg.budget || 1000,
            resize: cfg.resize || 224,
            available_architectures: cfg.available_architectures || []
        };
        sampleFilterView.applyServerConfig(cfg);
        if (state.classColors instanceof Map) {
            state.classColors.clear();
        }
        state.configUpdated = false;
    }

    await loadConfigFromServer();
    const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');
    const sampleInfoView = sampleInfoContainer ? new SampleInfoView({ container: sampleInfoContainer, state }) : null;

    // Overlay slider controls prediction/annotation overlay strength (0-100%)
    const overlaySlider = document.getElementById('overlay-slider');
    const overlayValue = document.getElementById('overlay-slider-value');
    const overlayToggle = document.getElementById('overlay-toggle');
    if (overlaySlider) {
        const applyOverlay = () => {
            const pct = Number(overlaySlider.value || 0);
            if (overlayValue) overlayValue.textContent = `${pct}%`;
            imageView.overlayAlpha = Math.max(0, Math.min(1, pct / 100));
        };
        overlaySlider.addEventListener('input', applyOverlay);
        applyOverlay();
    }
    if (overlayToggle) {
        overlayToggle.addEventListener('change', () => {
            imageView.overlayVisible = overlayToggle.checked;
        });
        // Set initial state
        imageView.overlayVisible = overlayToggle.checked;
    }

    const classesView = new PointsClassesView(classPanel, selectClassWorkflow, state);
    const statsView = new StatsView(api, classesView);
    const trainingCurveView = new TrainingCurveView(api);
    // Use segmentation-specific architectures for points annotation
    const segmentationArchitectures = ['small', 'small+', 'base', 'large'];
    aiControlsView = new AIControlsView(api, state, segmentationArchitectures);
    // No strategy selection in segmentation frontend

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

    function updateMaskAnnotationToggle() {
        if (!maskAnnotationToggle) return;
        const hasAnnotationMask = Array.isArray(state.currentMaskAnnotations) && state.currentMaskAnnotations.length > 0;
        const predictionMap = state.currentMaskPredictions && typeof state.currentMaskPredictions === 'object'
            ? state.currentMaskPredictions
            : {};
        const predictionClasses = Object.keys(predictionMap);
        const hasPrediction = predictionClasses.length > 0;
        const isBusy = state.workflowInProgress || state.isAcceptingMask || state.isRemovingMask;

        maskAnnotationToggle.checked = hasAnnotationMask;
        maskAnnotationToggle.indeterminate = isBusy;

        let reason = '';
        if (isBusy) {
            reason = 'Mask operation in progress.';
        } else if (!state.currentSampleId) {
            reason = 'No sample loaded.';
        } else if (!hasAnnotationMask && !hasPrediction) {
            reason = 'No mask prediction available.';
        }

        maskAnnotationToggle.disabled = Boolean(reason);
        maskAnnotationToggle.title = reason;
    }

    updateMaskAnnotationToggle();

    function beginWorkflow() {
        state.workflowInProgress = true;
        setInteractiveEnabled(false);
        updateMaskAnnotationToggle();
    }
    function endWorkflow() {
        state.workflowInProgress = false;
        setInteractiveEnabled(true);
        updateMaskAnnotationToggle();
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

        updateMaskAnnotationToggle();

        // You can also trigger visual updates here if needed
        // For example, update cursor style or selected point color preview
    }
    // No strategy view in segmentation; nothing to sync

    async function annotateWorkflow(sampleId, className) {
        if (state.workflowInProgress) return; // guard
        beginWorkflow();
        try {
            await updateConfigIfNeeded();
            await api.annotateSample(sampleId, className);
            await loadSampleAndContext(null, null);
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

    // No next-image strategy: always use default backend selection
    async function loadNextImage() {
        await loadSampleAndContext(null, null);
        // Note: No prediction display for points annotation
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

        // Update class colors when classes change
        state.config.classes.forEach(className => {
            getClassColor(className); // Ensures color is assigned
        });

        classesView.render();
        if (aiControlsView) {
            aiControlsView.render(state.config);
        }
        sampleFilterView.render();
    }
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    async function savePointsForCurrentImage() {
        if (!state.currentSampleId) return;
        try {
            const points = imageView.points || [];
            console.log('savePointsForCurrentImage:', { sampleId: state.currentSampleId, points });
            if (!points || points.length === 0) {
                // Explicitly clear points for this image when the set is empty
                await api.clearPoints(state.currentSampleId);
            } else {
                await api.savePointAnnotations(state.currentSampleId, points);
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
            const prevSample = await api.loadSamplePrev(state.currentSampleId);
            if (!prevSample) {
                alert('No previous image available');
            } else {
                await loadSampleAndContext(prevSample, null);
            }
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
            let nextSample = null;

            // Try to get next sample if we have a current sample
            if (state.currentSampleId) {
                nextSample = await api.loadSampleNext(state.currentSampleId);
            }

            // If no next sample found, load a new one using strategy params
            if (!nextSample) {
                await loadSampleAndContext(null, null);
            } else {
                await loadSampleAndContext(nextSample, null);
            }
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
        state.isAcceptingMask = false;
        state.isRemovingMask = false;
        if (sampleInfoView) {
            sampleInfoView.update({ sampleId: state.currentSampleId, filepath });
        }
        imageView.loadImage(imageUrl, filepath);
        imageView.clearPoints(); // Clear existing points when loading new image

        state.currentMaskPredictions = null;
        state.currentMaskAnnotations = null;

        // Load existing point and mask annotations from backend
        let maskAnnotations = [];
        try {
            const annotationsData = await api.getAnnotations(sampleId);
            const annotationsList = Array.isArray(annotationsData.annotations) ? annotationsData.annotations : [];
            const pointAnnotations = annotationsList.filter(ann => ann.type === 'point');
            maskAnnotations = annotationsList.filter(ann => ann.type === 'mask' && ann.mask_url);

            // Add existing points to the image view
            pointAnnotations.forEach(ann => {
                const color = getClassColor(ann.class);
                const x = (typeof ann.col01 === 'number') ? ann.col01 / 1_000_000 : (ann.col ?? 0);
                const y = (typeof ann.row01 === 'number') ? ann.row01 / 1_000_000 : (ann.row ?? 0);
                imageView.addExistingPoint(x, y, ann.class, color);
            });

        } catch (error) {
            console.error('Failed to load existing annotations:', error);
        }

        if (maskAnnotations.length > 0) {
            const annotationMaskMap = {};
            maskAnnotations.forEach(ann => {
                if (ann && ann.class && ann.mask_url) {
                    annotationMaskMap[ann.class] = { url: ann.mask_url, source: 'annotation' };
                }
            });
            if (Object.keys(annotationMaskMap).length > 0) {
                await loadMaskAssets(annotationMaskMap, { source: 'annotation' });
            } else if (imageView) {
                imageView.setMaskOverlays({ overlays: null, colors: null });
            }
            state.currentMaskAnnotations = maskAnnotations;
            state.currentMaskPredictions = null;
        } else if (predictions && predictions.type === 'mask' && predictions.mask_map) {
            const meta = await loadMaskAssets(predictions.mask_map, { source: 'prediction' });
            state.currentMaskPredictions = meta;
            state.currentMaskAnnotations = null;
        } else if (imageView) {
            imageView.setMaskOverlays({ overlays: null, colors: null });
            state.currentMaskPredictions = null;
            state.currentMaskAnnotations = null;
        }

        updateMaskAnnotationToggle();

        await classesView.setCurrentSample(sampleId, filepath);
    }

    // DRY helper: push config, save points, fetch sample (or use provided),
    // load annotations, load masks, refresh stats/config, update buttons
    async function loadSampleAndContext(sampleDataOrNull, _unused) {
        await updateConfigIfNeeded();
        await savePointsForCurrentImage();
        let data = sampleDataOrNull;
        if (!data) {
            // No strategy: call API without query params
            data = await api.loadNextImage(null, null, null);
        }
        await loadSampleData(data);
        await getStatsAndConfig();
        updateNavigationButtons();
    }

    // Helper: convert 1-bit PNG mask to alpha mask (white->opaque, black->transparent)
    function toAlphaMask(img, width, height) {
        const tmp = document.createElement('canvas');
        tmp.width = width;
        tmp.height = height;
        const ctx = tmp.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height);
        let needsAlpha = false;
        for (let i = 0; i < imageData.data.length; i += 4) {
            if (imageData.data[i+3] < 255) {
                // Already has alpha
                return img;
            }
            if (imageData.data[i] > 200 && imageData.data[i+1] > 200 && imageData.data[i+2] > 200) {
                imageData.data[i+3] = 255;
                needsAlpha = true;
            } else {
                imageData.data[i+3] = 0;
                needsAlpha = true;
            }
        }
        if (!needsAlpha) return img;
        ctx.putImageData(imageData, 0, 0);
        const outImg = new window.Image();
        outImg.src = tmp.toDataURL();
        return outImg;
    }

    async function loadMaskAssets(maskMap, { source = null } = {}) {
        const overlays = {};
        const overlayColors = {};
        const metadata = {};
        const entries = Object.entries(maskMap || {});
        await Promise.all(entries.map(([cls, rawValue]) => new Promise((resolve) => {
            const info = (typeof rawValue === 'string') ? { url: rawValue } : (rawValue && typeof rawValue === 'object' ? { ...rawValue } : null);
            if (!info || !info.url) {
                resolve();
                return;
            }
            const img = new window.Image();
            img.onload = () => {
                const alphaImg = toAlphaMask(img, img.width, img.height);
                overlays[cls] = alphaImg;
                const meta = { url: info.url };
                const idVal = info.prediction_id ?? info.id;
                const tsVal = info.prediction_timestamp ?? info.timestamp;
                if (idVal != null && !Number.isNaN(Number(idVal))) {
                    meta.prediction_id = Number(idVal);
                }
                if (tsVal != null && !Number.isNaN(Number(tsVal))) {
                    meta.prediction_timestamp = Number(tsVal);
                }
                if (source) meta.source = source;
                if (info.source && !meta.source) meta.source = info.source;
                metadata[cls] = meta;
                resolve();
            };
            img.onerror = () => { console.warn('Failed to load mask asset', info.url); resolve(); };
            img.src = info.url;
        })));
        const overlayKeys = Object.keys(overlays);
        if (overlayKeys.length > 0) {
            for (const cls of overlayKeys) {
                overlayColors[cls] = getClassColor(cls);
            }
            imageView.setMaskOverlays({ overlays, colors: overlayColors });
        } else {
            imageView.setMaskOverlays({ overlays: null, colors: null });
        }
        return metadata;
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
    async function handleAcceptMaskPrediction() {
        if (!state.currentSampleId || state.workflowInProgress || state.isAcceptingMask || state.isRemovingMask) {
            return false;
        }
        const predictionMap = state.currentMaskPredictions && typeof state.currentMaskPredictions === 'object'
            ? state.currentMaskPredictions
            : {};
        const entries = Object.entries(predictionMap).filter(([cls, meta]) => cls && meta && typeof meta === 'object');
        if (entries.length === 0) {
            alert('No mask prediction available to save.');
            return false;
        }

        const missingMeta = [];
        const payload = entries.map(([cls, meta]) => {
            const predictionId = meta.prediction_id ?? meta.id ?? null;
            const predictionTimestamp = meta.prediction_timestamp ?? meta.timestamp ?? null;
            if (predictionId == null || predictionTimestamp == null) {
                missingMeta.push(cls);
            }
            return {
                className: cls,
                predictionId,
                predictionTimestamp,
            };
        }).filter(item => item.predictionId != null && item.predictionTimestamp != null);

        if (missingMeta.length > 0 || payload.length === 0) {
            const missingList = missingMeta.join(', ');
            alert(missingList
                ? `Cannot save masks: missing prediction metadata for ${missingList}.`
                : 'Cannot save masks: missing prediction metadata.');
            return false;
        }

        // Keep UI selection aligned with the first accepted class when possible.
        const primaryClass = state.selectedClass && predictionMap[state.selectedClass]
            ? state.selectedClass
            : payload[0].className;
        state.selectedClass = primaryClass;
        if (classesView && typeof classesView.setSelectedClass === 'function' && primaryClass) {
            classesView.setSelectedClass(primaryClass);
        }

        state.isAcceptingMask = true;
        updateMaskAnnotationToggle();
        beginWorkflow();
        try {
            await api.acceptMaskPrediction(state.currentSampleId, payload);
            const refreshed = await api.loadSample(state.currentSampleId);
            await loadSampleAndContext(refreshed, null);
            return true;
        } catch (err) {
            if (err && err.status === 409) {
                alert('Mask prediction changed while saving. Reloading latest version.');
                try {
                    const refreshed = await api.loadSample(state.currentSampleId);
                    await loadSampleAndContext(refreshed, null);
                } catch (reloadErr) {
                    console.error('Failed to reload sample after prediction change:', reloadErr);
                }
            } else {
                console.error('Failed to accept mask predictions:', err);
                const message = err && err.message ? err.message : 'Unknown error';
                alert(`Failed to save masks: ${message}`);
            }
            return false;
        } finally {
            state.isAcceptingMask = false;
            endWorkflow();
            updateMaskAnnotationToggle();
        }
    }

    async function handleRemoveMaskAnnotation() {
        if (!state.currentSampleId || state.workflowInProgress || state.isAcceptingMask || state.isRemovingMask) {
            return false;
        }
        const annotations = Array.isArray(state.currentMaskAnnotations)
            ? state.currentMaskAnnotations.filter(ann => ann && ann.type === 'mask')
            : [];
        if (annotations.length === 0) {
            return true;
        }

        state.isRemovingMask = true;
        updateMaskAnnotationToggle();
        beginWorkflow();
        try {
            await api.deleteMaskAnnotations(state.currentSampleId);
            const refreshed = await api.loadSample(state.currentSampleId);
            await loadSampleAndContext(refreshed, null);
            return true;
        } catch (err) {
            console.error('Failed to delete mask annotation:', err);
            const message = err && err.message ? err.message : 'Unknown error';
            alert(`Failed to remove mask annotations: ${message}`);
            return false;
        } finally {
            state.isRemovingMask = false;
            endWorkflow();
            updateMaskAnnotationToggle();
        }
    }

    if (maskAnnotationToggle) {
        maskAnnotationToggle.addEventListener('change', async () => {
            if (maskAnnotationToggle.disabled) return;
            const hadAnnotation = Array.isArray(state.currentMaskAnnotations) && state.currentMaskAnnotations.length > 0;
            const wantsAnnotation = maskAnnotationToggle.checked;
            if (wantsAnnotation === hadAnnotation) {
                updateMaskAnnotationToggle();
                return;
            }

            maskAnnotationToggle.disabled = true;
            maskAnnotationToggle.title = 'Mask operation in progress.';

            try {
                if (wantsAnnotation && !hadAnnotation) {
                    const success = await handleAcceptMaskPrediction();
                    if (!success) {
                        maskAnnotationToggle.checked = false;
                    }
                } else if (!wantsAnnotation && hadAnnotation) {
                    const success = await handleRemoveMaskAnnotation();
                    if (!success) {
                        maskAnnotationToggle.checked = true;
                    }
                }
            } finally {
                updateMaskAnnotationToggle();
            }
        });
    }
    function initKeyboard(api) {
        const hk = new Hotkeys();
        const guard = (fn) => (e) => { if (state.workflowInProgress) return; fn(e); };
        const undoPoint = () => {
            const lastPoint = imageView.getLastPoint();
            if (lastPoint) {
                imageView.removeLastPoint();
            }
        };
        const clearPoints = () => { imageView.clearPoints(); };
        const selectIndex = (idx) => {
            if (idx < state.config.classes.length) {
                const className = state.config.classes[idx];
                selectClassWorkflow(state.currentSampleId, className);
            }
        };
        const toggleOverlay = () => {
            if (overlayToggle) {
                overlayToggle.checked = !overlayToggle.checked;
                overlayToggle.dispatchEvent(new Event('change'));
            }
        };
        hk
          .bind('ctrl+e', guard(() => api.exportDB()))
          .bind('meta+e', guard(() => api.exportDB()))
          .bind('ctrl+arrowleft', guard(() => navigatePrev()))
          .bind('meta+arrowleft', guard(() => navigatePrev()))
          .bind('ctrl+arrowright', guard(() => navigateNext()))
          .bind('meta+arrowright', guard(() => navigateNext()))
          .bind('b', guard(() => navigatePrev()))
          .bind('n', guard(() => navigateNext()))
          .bind('m', guard(() => toggleOverlay()))
          .bind('ctrl+u', guard(() => undoPoint()))
          .bind('meta+u', guard(() => undoPoint()))
          .bind('ctrl+c', guard(() => clearPoints()))
          .bind('meta+c', guard(() => clearPoints()))
          .bind('1', guard(() => selectIndex(0)))
          .bind('2', guard(() => selectIndex(1)))
          .bind('3', guard(() => selectIndex(2)))
          .bind('4', guard(() => selectIndex(3)))
          .bind('5', guard(() => selectIndex(4)))
          .bind('6', guard(() => selectIndex(5)))
          .bind('7', guard(() => selectIndex(6)))
          .bind('8', guard(() => selectIndex(7)))
          .bind('9', guard(() => selectIndex(8)))
          .bind('0', guard(() => selectIndex(9)))
          .attach();
    }
    initKeyboard(api);
    try {
        await loadNextImage(); // loadSampleAndContext already refreshes stats/config
        updateNavigationButtons(); // Initialize button states
    } catch (e) {
        console.error('Failed to initialize application:', e);
    }
});
