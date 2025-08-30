# General

| User Action | Consequence | Details |
| --- | --- | --- |
| Open app (router) | See task options | Choose `classification` or `segmentation`. |
| Select task | Images available | Backend loads images after DB init; user does not manage data. |
| Manage classes | Class list updates | Add/remove classes for this session; numeric keys map to visible classes. |
| View training | Metrics visible | Live validation accuracy and curves. |
| Edit training config | New config scheduled | Changes apply next cycle or on restart; current model/version shown. |
| Export | File downloaded | Annotations exported only (predictions excluded in v1). |



# Classification

| User Action | Consequence | Details |
| --- | --- | --- |
| Click class / press number | Saved and Next | Saves label immediately and advances per selected strategy; uses current class list/hotkeys. |
| Change strategy | Next selection changes | Strategies (UX names): `sequential` (id order), `random` (uniform), `minority min-prob` (lowest-probability prediction for the minority class; fallback random if none), `last class max-prob` (highest-probability prediction for the last annotated class; fallback random if none), `specific class` (requires class selection). Mapping to API params: `minority min-prob` → `strategy=minority_frontier`; `specific class` → `strategy=specific_class&class=<selected>`; `last class max-prob` uses `strategy=specific_class&class=<last_annotated_class>`. |

# Segmentation

| User Action | Consequence | Details |
| --- | --- | --- |
| Select class | Target class set | Applies to subsequent points/edits; numeric hotkeys select classes. |
| Add/remove points | Points updated | Point annotations update for the selected class; no mask update. |
| Overlay slider | Viewer updates | Adjust visibility/strength of predictions and existing annotations. |
| Delete points | Current image reset | Remove all point annotations for the current image. |
| Next | Advance to Next | Uses current strategy (same set as classification); `prev/next` are deterministic and return 404 at list boundaries (no wrap). |
