
# The ultimate image annotation tool

Annotate images fast with human-in-the-loop learning for annotation.

Run with: `uvicorn src.backend.main:app`

## Vision
Annotating images is a bottleneck, this eases the process. There are three big problems in computer vision, classification, detection, and segmentation (instance / semantic, see later). They are all about classifying, one classifies images, another bounding boxes, and another classifies pixels. 

HITL learning for annotation is best illustrated in the object detection case: the AI predicts bounding boxes for each class and the user corrects the bounding boxes and/or their classes. As the user annotates, the AI learns, and the user makes fewer and fewer corrections. 

For classification it's a little different because the cost of confirmation and the cost of correction  are almost the same. Speeding up annotation in this case is harder. Also, active learning might be preferred. The ways to do classification right are two: 1. active learning, 2. faster annotation. Active learning involves more or less fancy methods, of which the best is usually maximum uncertainty of an ensemble. Then we have random sampling, and then we have sampling the least frequent class. I like the latter a lot. One issue we have is that we can't do the latter if we save only the predicted class, can we? Maybe if we take the most confident of the least represented class or the least confident of the second least represented class. Or maybe always the least confident of the least represented class, that we could do. So let us fix two methods: random and active (least conf least freq). For quick annotation, the strategy is to batch. We can batch in a sequence or we can batch in the UI. To keep it simple, we will batch in a sequence. This strategy is basically about choosing the sample with highest probability of belonging to the last annotated class.
In short, we will implement two methods: least confidence of least frequent class, and most confident of the last annotated class.


## DESIGN

### Frontend:
- Image, classes, performance and training plots, config button. Other buttons and text to custom visualization (image name, progress bar, reset view, undo, disable zoom)

### Backend:
#### API
PUT config (config), returns success or not
GET next (strategy), returns image payload, id, prediction or annotation <- strategy can be "randoms", "actives" or <classname>. If image can't be sent with the rest then we need to divide this in two
GET sample (id), returns image payload, prediction or annotation
POST annotate (id, annotation), returns success or not
DELETE annotate (id), returns success or not
GET stats (), returns stats

### Database
- SQLite for storing image paths, annotations, predictions, and config / stats

## Docs

### Backend API
- **/api/sample?id=IMAGE_ID**: Returns the image file for the given IMAGE_ID. IMAGE_ID must match one of the available image paths.
- **/api/ids**: Returns a JSON array of all available image IDs (file paths).
- The backend is built with Starlette and serves the frontend as static files.

### Frontend Features
- **Image Viewer**: Displays images on a canvas with support for multiple images.
- **Navigation**: Use Prev/Next buttons to navigate through images. The current image ID and position are shown below the canvas.
- **Loading State**: A loading overlay appears while images are being fetched.
- **Zoom & Pan**: Use mouse wheel to zoom (centered on cursor), and drag to pan the image.
- **Reset View**: The Reset View button restores the image to fit the viewport.
- **Responsive Canvas**: The canvas resizes with the window.

### Usage
1. Start the backend server (see main.py for details, typically with `uvicorn`).
2. Open the frontend in your browser (served at `/`).
3. Navigate images using the Prev/Next buttons. Zoom and pan as needed.
4. The image list is refreshed after each navigation to stay up-to-date.

### Notes
- Image IDs are full file paths as defined in the backend.
- If you change an image file, your browser may cache the old version. Refresh or clear cache if needed.

## DONE
- image loader (backend, single image)
- image viewer (single image)
- image viewer (multiple images)
    [x] change the sample api to get by id
    - add navigation
        [x] prev next buttons
        [x] get requests right
        [x] block requests
    [x] update the frontend to display the image ID
    [x] make the frontend request the list everytime we navigate (after navigating)
    [x] make the frontend display a spinner if the image is loading
- image viewer (panning)
- image viewer (zooming)
- documentation
- draft database (multiple images, sqlite)
- integrate database into app (multiple images, sqlite)
- add class handling in the frontend (add class)
- write class to backend
- load class from backend

## TODO
- add active learning
- add quick annotation
- resume with the classes so far
- add hitl
    [x] add fastai training of classifier from the database
    [x] write predictions to the database
    [x] show predictions in the frontend
    [x] fix fastai
    [] add preprocessing to be edited by user
- support class removal 
- image loader (.zip)