# AGENTS — Guidance for AI coding agents working on bLUe

Checklist for an agent starting work here
- Read the high-level entrypoint: `bLUe.py` (app bootstrap, main menu routing).
- Inspect GUI / window globals in `bLUeTop/Gui.py` and `bLUeTop/settings.py` (platform config and flags).
- Understand the layer pipeline in `bLUeTop/MarkedImg.py` (class QLayer and applyToStack / getCurrentMaskedImage).
- Find where new features are registered: `layerScripting` in `bLUe.py` (how graphic forms + execute wrappers are wired).
- Edit only the minimal files required; preserve public APIs and UI wiring.

Quick architecture summary (big picture)
- Entry point: `bLUe.py` — creates the QApplication via `bLUeTop.Gui`, sets up the main window and menu handlers.
- GUI package: `bLUeTop/` — contains GUI forms, the central `MarkedImg.py` (layer model), `versatileImg.py` (image containers) and many graphics forms under `bLUeTop/graphics*` used as layer editors.
- Core image algorithms: `bLUeCore/` — LUTs, interpolation, filters and numerical helpers.
- GUI helpers and image buffer abstractions: `bLUeGui/` — QImage <-> ndarray helpers, dialogs, and logging (`bLUeGui/logginit.py`).
- ML & AI: `bLUeNN/` (optional Torch models) and `bLUeAI/segmasks.py` (conversational segmentation UI integration).

Important conventions and project-specific patterns
- Layer-first architecture: image editing is implemented as a stack of QLayer instances (see `bLUeTop/MarkedImg.py`). A 'presentation' layer composes the stack for display.
- The "execute wrapper" pattern: many layers are generic QLayer instances whose processing function is attached at runtime by assigning `layer.execute = lambda ...`. See `layerScripting()` in `bLUe.py` for many examples (e.g. auto 3D LUT, curves, RAW develop).
- Role strings and lightweight typing: layers are distinguished using a `role` string (examples: 'RAW', '3DLUT', 'DRW', 'TXT', 'CLONING'). Use `layer.isRawLayer()` / `is3DLUTLayer()` helpers rather than adding new heavyweight subclasses unless necessary.
- Preview / Hald modes: the app supports three representations — full image, preview/thumb and hald (identity LUT) — selected via flags `parentImage.useThumb` and `parentImage.useHald`. Many methods branch on those flags (e.g. `QLayer.getCurrentImage()`).
- Channel order and buffer semantics: code frequently uses OpenCV-style BGR when manipulating raw buffers. You will often see `[..., ::-1]` when converting between QImageBuffer (RGB(A)) and algorithms expecting BGR — keep an eye on that to avoid color bugs.
- Masking: a layer's mask is a QImage stored on the layer (`layer.mask`) and controlled with `maskIsEnabled` / `maskIsSelected`. Mask updates generally require `layer.updatePixmap()` and `layer.applyToStack()` to propagate changes.
- State persistence: .blu files are written using TIFF / ImageJ metadata. Layer states are pickled into TIFF tags and restored by `loadImage()` (see `bLUe.py`). The loader uses `restricted_loads` to avoid unsafe unpickling — be cautious when changing serialization formats.
- Parallel interpolation: 3D LUT interpolation may use a multiprocessing pool controlled by `bLUeTop/settings.py` flags `USE_POOL` and `POOL_SIZE`. On Windows the code calls `multiprocessing.freeze_support()` in `bLUe.py` — keep this when adding multiprocessing.

Developer workflows & essential commands
- Install deps: create a venv and run
  ```powershell
  python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
  ```
- Edit platform config: update `config_win.json` (Windows) or `config.json` (Linux/macOS). `bLUeTop/settings.py` reads these at startup.
- Run the app (from project root):
  ```powershell
  python bLUe.py
  ```
- Logs: runtime logs go to `log.txt` (rotating handler configured in `bLUeGui/logginit.py`). Use that file for runtime traces and uncaught exceptions (the logger name is 'blue').

How to add a new processing layer (practical recipe)
1. Create a graphics form in `bLUeTop/` (use existing `graphics*` forms as templates).
2. If persistent GUI state is required, implement `__getstate__`/`__setstate__` on the form so it can be pickled in .blu files (many forms already do this).
3. Add the new form import and a branch to `layerScripting()` in `bLUe.py`. Use `window.label.img.addAdjustmentLayer(...)` to add the layer instance.
4. Attach the processing function by assigning `layer.execute = lambda l=layer, pool=None: l.tLayer.<your_method>(...)` or implement a `QLayer` subclass and set `layerType=` when adding.
5. If the layer is compatible with hald/3D LUT exports, set `layer.haldUsable = True`.
6. Wire a menu action (menu XML/UI is created in `bLUeTop/Gui.py` / .ui assets); use the same action naming conventions as `layerScripting()`.

Integration points & external dependencies
- Native tools: ExifTool is required and its path is configured in `config*.json` (the code uses `bLUeTop/settings.py` to pick platform paths). When frozen (bundled) the code expects `EXIFTOOL_PATH_BUNDLED`.
- Optional ML: Torch-based auto 3D LUT requires PyTorch (`bLUeNN`) and is gated by runtime check `HAS_TORCH` (in `bLUeTop/settings.py`). google-genai support is also optional and gated by `HAS_GENAI`.
- File I/O: .blu uses TIFF / ImageJ metadata via `tifffile`. Loading/saving layers relies on pickled states inside metadata tags — changing that format requires updating both writer and `loadImage()`'s reading logic.

Useful files to inspect first (quick pointers)
- `bLUe.py` — app entry, menus, layerScripting, load/save flow
- `bLUeTop/MarkedImg.py` — QLayer, applyToStack(), masking and blending
- `bLUeTop/versatileImg.py` — image container helpers and color conversions
- `bLUeTop/settings.py` + `config.json` / `config_win.json` — platform flags and feature switches
- `bLUeGui/logginit.py` — logging setup (`log.txt`)
- `bLUeCore/` — numeric algorithms and LUT support
- `bLUeNN/` — ML models used by Auto 3D LUT
- `bLUeAI/segmasks.py` — conversational segmentation UI integration

Short maintenance notes for agents
- Preserve UI wiring: many GUI objects are referenced globally via `bLUeTop.Gui.window`. Avoid large refactors that change these global access patterns.
- Small, localized changes are preferred. Most runtime behavior is configured by assigning callables to layer.execute — changing that pattern affects many layers.
- When changing serialization of layer forms, update both saving (writer) and loading (`bLUe.py::loadImage`) and include backwards-compatibility handling (the loader already catches old-format errors and shows warnings).

If anything in this AGENTS.md is unclear, open the files listed above and search for the keywords shown (e.g. `layerScripting`, `QLayer`, `applyToStack`, `useThumb`, `useHald`).

