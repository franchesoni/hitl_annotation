import { API } from '/shared/js/api.js';
import { ImageView } from '/shared/views/imageView.js';
import { PointsClassesView } from './views/pointsClassesView.js';
import { StatsView } from '/shared/views/statsView.js';
import { AIControlsView } from '/shared/views/aiControlsView.js';
import { TrainingCurveView } from '/shared/views/trainingCurveView.js';

document.addEventListener('DOMContentLoaded', async () => {
        // -----------------------------------------------------------
        // ----------  STATE  ----------------------------------------
        // -----------------------------------------------------------
        const state = {
                config: { classes: [], aiShouldBeRun: false, architecture: 'small', budget: 1000, sleep: 0, resize: 224 },
                configUpdated: false,
                workflowInProgress: false,
                currentImageFilepath: null,
                currentSampleId: null, // Track current sample ID for navigation
                currentStats: null,  // Store current stats for optimization
                selectedClass: null,
                classColors: new Map(), // Map class names to colors
        };

        // Generate distinct colors for classes
        function generateClassColor(index) {
                const colors = [
                        '#FF6B6B', // red
                        '#4ECDC4', // teal
                        '#45B7D1', // blue
                        '#96CEB4', // green
                        '#FFEAA7', // yellow
                        '#DDA0DD', // plum
                        '#98D8C8', // mint
                        '#F7DC6F', // light yellow
                        '#BB8FCE', // light purple
                        '#85C1E9', // light blue
                        '#F8C471', // orange
                        '#82E0AA', // light green
                        '#F1948A', // light red
                        '#85929E', // gray
                        '#D7BDE2', // lavender
                ];
                return colors[index % colors.length];
        }

        // Get color for a class, assigning one if not exists
        function getClassColor(className) {
                if (!state.classColors.has(className)) {
                        const classIndex = state.config.classes.indexOf(className);
                        const color = generateClassColor(classIndex);
                        state.classColors.set(className, color);
                }
                return state.classColors.get(className);
        }
        
        // -----------------------------------------------------------
        // ----------  COMPONENTS  -----------------------------------
        // -----------------------------------------------------------
        const leftPanel = document.querySelector('.left-panel');
        const classPanel = document.querySelector('#class-manager');
        const api = new API();
        await loadConfigFromServer();
        const imageView = new ImageView(leftPanel, 'loading-overlay', 'c');
        
        // Set up point addition callback
        imageView.onPointAdd = async (point) => {
                console.log('Point added:', point);
                try {
                        await api.addPoint(state.currentSampleId, point.className, point.x, point.y);
                        console.log('Point saved to backend');
                } catch (error) {
                        console.error('Failed to save point:', error);
                        // Optionally remove the point from frontend if backend save failed
                        imageView.removeLastPoint();
                        alert('Failed to save point annotation');
                }
        };
        
        // Set up point removal callback
        imageView.onPointRemove = async (point, index) => {
                console.log('Point removed:', point, 'at index:', index);
                try {
                        await api.deletePoint(state.currentSampleId, point.x, point.y);
                        console.log('Point removed from backend');
                } catch (error) {
                        console.error('Failed to remove point:', error);
                        // Optionally re-add the point to frontend if backend delete failed
                        alert('Failed to remove point annotation');
                }
        };
        const classesView = new PointsClassesView(classPanel, selectClassWorkflow, state);
        const statsView = new StatsView(api, classesView);
        const trainingCurveView = new TrainingCurveView(api);
        // Use segmentation-specific architectures for points annotation
        const segmentationArchitectures = ['small', 'small+', 'base', 'large'];
        const aiControlsView = new AIControlsView(api, state, segmentationArchitectures);
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

        async function loadNextImage() {
                const { imageUrl, sampleId, filepath, labelClass, labelSource, labelProbability } =
                        await api.loadNextImage();
                state.currentImageFilepath = filepath;
                state.currentSampleId = parseInt(sampleId);
                
                imageView.loadImage(imageUrl, filepath);
                imageView.clearPoints(); // Clear existing points when loading new image
                
                // Load existing point annotations from backend
                try {
                        const annotationsData = await api.getAnnotations(sampleId);
                        const pointAnnotations = annotationsData.annotations.filter(ann => ann.type === 'point');
                        
                        // Add existing points to the image view
                        pointAnnotations.forEach(ann => {
                                const color = getClassColor(ann.class);
                                imageView.addExistingPoint(ann.col, ann.row, ann.class, color);
                        });
                        
                        console.log(`Loaded ${pointAnnotations.length} existing points`);
                } catch (error) {
                        console.error('Failed to load existing annotations:', error);
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
                                        classes: cfg.classes || [],
                                        aiShouldBeRun: cfg.ai_should_be_run || false,
                                        architecture: cfg.architecture || 'small', // Default to 'small' for segmentation
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
                
                // Update class colors when classes change
                state.config.classes.forEach(className => {
                        getClassColor(className); // Ensures color is assigned
                });
                
                classesView.render();
                aiControlsView.render(state.config);
        }
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');

        async function navigatePrev() {
                if (state.workflowInProgress) return;
                if (!state.currentSampleId) {
                        alert('No current sample to navigate from');
                        return;
                }
                
                beginWorkflow();
                try {
                        const prevSample = await api.loadSamplePrev(state.currentSampleId);
                        if (prevSample) {
                                await loadSampleData(prevSample);
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
                        let nextSample = null;
                        
                        // Try to get next sample if we have a current sample
                        if (state.currentSampleId) {
                                nextSample = await api.loadSampleNext(state.currentSampleId);
                        }
                        
                        // If no next sample found, load a new one
                        if (!nextSample) {
                                await loadNextImage();
                                await getStatsAndConfig();
                        } else {
                                await loadSampleData(nextSample);
                        }
                        
                        updateNavigationButtons();
                } catch (e) {
                        console.error('Navigate next failed:', e);
                } finally {
                        endWorkflow();
                }
        }

        async function loadSampleData(sampleData) {
                const { imageUrl, sampleId, filepath } = sampleData;
                state.currentImageFilepath = filepath;
                state.currentSampleId = parseInt(sampleId);
                imageView.loadImage(imageUrl, filepath);
                imageView.clearPoints(); // Clear existing points when loading new image
                
                // Load existing point annotations from backend
                try {
                        const annotationsData = await api.getAnnotations(sampleId);
                        const pointAnnotations = annotationsData.annotations.filter(ann => ann.type === 'point');
                        
                        // Add existing points to the image view
                        pointAnnotations.forEach(ann => {
                                const color = getClassColor(ann.class);
                                imageView.addExistingPoint(ann.col, ann.row, ann.class, color);
                        });
                        
                        console.log(`Loaded ${pointAnnotations.length} existing points`);
                } catch (error) {
                        console.error('Failed to load existing annotations:', error);
                }
                
                await classesView.setCurrentSample(sampleId, filepath);
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
                                        // Remove from backend too
                                        api.deletePoint(state.currentSampleId, lastPoint.x, lastPoint.y).catch(err => {
                                                console.error('Failed to remove point from backend:', err);
                                        });
                                }
                        } else if (lowerCaseKey === 'c' && (e.ctrlKey || e.metaKey)) {
                                e.preventDefault();
                                // Clear all points
                                imageView.clearPoints();
                                // Clear from backend too
                                api.clearPoints(state.currentSampleId).catch(err => {
                                        console.error('Failed to clear points from backend:', err);
                                });
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