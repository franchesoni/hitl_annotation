## Overview

The frontend keeps a local config mirror that syncs with the backend on specific triggers, and all image navigation (next, prev, undo) follows defined API call sequences. This file specifies the order and conditions under which endpoints in `system_arch/api.md` are called.

## Shared Behavior

- Config I/O: Keep a local mirror. Read via `GET /api/config`; write via `PUT /api/config` (merge semantics; do not remove unspecified keys).
- Push policy:
  - Toggle run: When `ai_should_be_run` is set or unset, push the updated config immediately.
  - Image change: On navigation to a different image, push any pending config changes (e.g., class list updates) before fetching the next image; after the next image loads, refresh config via `GET /api/config` and apply it to the UI (hotkeys, visible classes).

## Image-Change Workflows (API Order)

- Next (strategy-driven):
  1) If pending config changes exist, `PUT /api/config` (merge-only).
  2) If there are unsaved annotations for the current image, submit them before leaving:
     - Classification: when Next is triggered by a class selection, `PUT /api/annotations/{id}` with `[ { type: "label", class, timestamp } ]`.
     - Segmentation: `PUT /api/annotations/{id}` with a list of `{ type: "point", ... }` to batch-save points.
  3) `GET /api/samples/next` (include strategy params if applicable).
  4) Parse prediction headers.
  5) For segmentation, `GET /api/annotations/{id}` to load prior annotations; then, if mask predictions are advertised, fetch mask assets listed in `X-Predictions-Mask`.
  6) `GET /api/config` and `GET /api/stats` to refresh session state; update local mirror and UI.

- Prev (deterministic back):
  1) Same pre-leave steps as Next (config push, segmentation batch-save if needed).
  2) `GET /api/samples/{id}/prev`.
  3) Repeat steps 4–6 from Next.

- Undo (return to previous image):
  1) Update config first: `PUT /api/config` if there are any pending changes.
  2) Load the previous image deterministically via `GET /api/samples/{id}/prev`.
  3) Refresh session state with `GET /api/config` and `GET /api/stats`.

## Classification View

- Save workflow (API order):
  1) `PUT /api/annotations/{id}` with `[ { type: "label", class, timestamp } ]`.
  2) Follow the Next workflow sequence above (config push if needed → `GET /api/samples/next` → headers/annotations as applicable → refresh config/stats).
 - Strategy parameters:
   - `sequential`, `random`, `minority_frontier`: no additional params.
   - `specific_class`: include the selected class parameter from the UI.
   - Last-class (frontend convenience): track the last successfully annotated class locally and call `/api/samples/next` with `strategy=specific_class&class=<last_annotated_class>`. If no last class exists yet, fall back to a default (e.g., `sequential`).

## Segmentation View

- Load on entry: After fetching an image, parse prediction headers (`X-Predictions-*`), then call `GET /api/annotations/{id}` to seed local points. If `X-Predictions-Type=mask`, fetch each mask asset after annotations load.
- Edit locally: Maintain a per-class list of points; do not write on every edit.
- Save on leave (API order): On navigation, push pending config if any → `PUT /api/annotations/{id}` with all `{type:"point", col01, row01, [class]}` for the current image (overwrite-by-type semantics; `col01`/`row01` are integers in ppm of image width/height, 0..1,000,000) → proceed to image fetch per the chosen workflow.
- Overwrite-by-type reminder: The server replaces all existing annotations of any `type` included in the payload for that sample with exactly those provided. To preserve other types, omit them from the payload. Batch and send the complete list for each included `type`; avoid per-edit writes that could unintentionally overwrite concurrent edits.
- Delete: Use `DELETE /api/annotations/{id}` with optional `type=point` to clear points for the current image; omit `type` to clear all types.
- Rendering note: Apply class colors and composite overlays; resize masks to match the displayed image dimensions before blending.

## Status & Feedback

- Loading: Show inline loading states for image fetches and saves.

## Live Accuracy Slider

- Purpose: Lets users view a windowed live accuracy computed over the most recent annotation events.
- Backend coordination:
  - On label save (`PUT /api/annotations/{id}` with `type="label"`), backend looks up the most recent label prediction for that sample and writes a live-accuracy point as a normal curve row into the `curves` table (curve_name=`live_accuracy`) via `store_live_accuracy(sample_id, is_correct)` where `is_correct ∈ {1.0, 0.0}`.
  - On stats fetch (`GET /api/stats`), backend returns available curves, including the standard curve named `live_accuracy` as an ordered list of `{value, timestamp}`, plus counts and any training curves.
- Frontend behavior:
  - UI provides a slider `0–100%` representing the window of most recent points to include.
  - When the slider changes, the frontend fetches fresh stats (or uses provided stats), takes the last `ceil(N * pct/100)` points from `live_accuracy_points`, and derives: `tries` (count), `correct` (sum of value==1.0), and `accuracy = correct/tries`.
  - Display is updated inline without blocking other interactions; no server call is made on slider drag beyond the periodic `GET /api/stats` refresh triggered by the view.
  - After each annotation, the usual flow runs: label save → Next workflow → `GET /api/stats`; the new live-accuracy point is then reflected in the slider’s windowed computation.

## Accessibility & Responsiveness

- Keyboard: All primary actions (class selection, next) are keyboard-accessible.
- Focus: Maintain sensible focus order; avoid stealing focus on image load.
- Layout: Works across common desktop resolutions without horizontal scrolling.

### Class Names (Frontend Validation)
- Only accept class names that are safe to append to filenames. Allowed pattern: `^[A-Za-z0-9_-]+$` (letters, digits, underscore, hyphen).
- No spaces, slashes, backslashes, quotes, or other special characters; reject and prompt the user to edit the name.
- Recommended max length: 64 characters; trim surrounding whitespace before validating.
- Rationale: class names may be appended to image filenames before the extension for derived artifacts; unsafe characters can break paths.


