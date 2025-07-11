
# The ultimate image annotation tool

Annotate images fast with human-in-the-loop learning for annotation.

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

## TODO
- image loader (.zip)
- database (multiple images, sqlite)