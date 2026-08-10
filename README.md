# Blender Synthetic Dataset Generator

A Blender/Cycles script that renders a synthetic COCO-style object detection + 6D pose dataset. For each generated image it randomizes the camera pose, object poses, lighting, background, and applies optional image degradation (blur/noise/contrast) to mimic a real camera (tuned here for an Meta Quest 3 camera). Output is compatible with a BOP/COCO (`train_pbr/.../rgb`, `annotations/instances_train.json`, `models/`, `camera.json`).

## What it generates

- **RGB renders** at a configurable resolution (default 1280x960, matching Quest camera intrinsics).
- **COCO annotations** (`instances_train.json`) with 2D bounding boxes, per-object category, and **6D pose** (rotation `R` + translation `T`, in the OpenCV camera convention) for every visible object.
- **Exported 3D models** (`.ply`) and per-model bounding box / diameter info (`models_info.json`).
- **Camera intrinsics** (`camera.json`).

## Project setup (in Blender)

1. **Blender version:** developed against **Blender 4.x** with the Cycles render engine.
2. **Open/create a `.blend` file** and add the objects you want to detect.
3. **Name each target object `TargetObject...`** (e.g. `TargetObject_bottle`, `TargetObject_mug`). The script auto-discovers every object whose name starts with `TargetObject` and sorts them alphabetically — this order defines the COCO `category_id`. Any object not named this way (and not the camera) gets **deleted** by `setup_scene()`, so keep your scene clean and only leave target objects + camera in it.
4. Objects should be roughly centered at the world origin with a sensible real-world scale (meters) — the script exports them at `base_scale_for_export` and randomizes location within a small radius around the origin for rendering.
5. **Background images (optional):** put a folder of background photos on disk and point `background_images_dir` at it. Images are sampled per run and used as HDRI-style environment backgrounds (`use_background_images`, `background_image_prob`).
6. **GPU rendering:** the script auto-detects and enables the best available Cycles backend in this order: `OPTIX` (RTX) → `CUDA` → `HIP` (AMD) → `METAL` (Mac), falling back to CPU with a warning if none are found. In practice, you may still need to manually set a couple of things in the project:
   - Switch the scene's render engine to **Cycles** yourself (Render Properties > Render Engine), even though the script also sets `scene.render.engine = 'CYCLES'` in code.
   - Enable the GPU device(s) in Blender's **Preferences > System > Cycles Render Devices** (and select the right backend there) so the auto-detection in the script actually finds them.
   
7. **Edit `CONFIG` at the top of the script** before running:
   - `output_dir`: where the dataset is written.
   - `num_images` / `start_index`: how many images to render, and the starting image index (useful for resuming/parallelizing).
   - `use_quest_intrinsics`: set to hardcoded Oculus Quest camera intrinsics (scaled to `image_width`/`image_height`), **it is necessary to change to the preferred parameters**!
   - `camera_distance_range`, `rotation_range`, `min_object_distance`: scene randomization ranges.
   - `degradation_prob`: probability of applying the image degradation (extra blur, noise, darker/low-contrast) per image, to simulate the real world envirnonment.
   - `num_lights`, `light_energy_range`, `cycles_samples`: lighting/render quality.
   - `cleanup_interval`: how often to purge orphaned mesh/image data blocks and checkpoint-save the COCO JSON (important for very large runs — this script is meant for tens of thousands of images).

## Running it

Paste/open the script in Blender's **Scripting** tab and press **Run**, or run it headless:

```bash
blender your_scene.blend --background --python datagen.py
```

### ⚠️ Windows note: run from the command line, not the Text Editor

On Windows, running this script from inside Blender's built-in **Text Editor / Scripting tab** (via the "Run Script" button) caused Blender to **crash** partway through generation. Launching Blender **from the command line** with the script passed via `--python` was stable and didn't crash:

```powershell
"C:\Program Files\Blender Foundation\Blender 4.x\blender.exe" "D:\path\to\your_scene.blend" --background --python "D:\path\to\datagen.py"
```

Running in `--background` mode also skips the GUI redraw overhead and lets the console print progress (`Generated i/N`) directly, which makes it easier to spot where a crash occurred if it happens again.

## Output structure

```
<output_dir>/
├── camera.json                     # camera intrinsics (fx, fy, cx, cy, width, height)
├── models/
│   ├── obj_000001.ply, ...         # exported target object meshes (mm units)
│   └── models_info.json            # bbox min/size + diameter per model, keyed by id
├── annotations/
│   └── instances_train.json        # COCO-style images/annotations/categories, with R/T pose per annotation
└── train_pbr/
    └── 000000/
        └── rgb/
            └── 000000.png, ...     # rendered images
```

## Notes / gotchas

- Objects are randomly sampled per image (`random.sample`, 1..N of them), so not every image contains every target object.
- The COCO JSON is checkpointed every `cleanup_interval` images, so a crash mid-run only loses progress since the last checkpoint, not the whole dataset — worth lowering this value on Windows given the crash issue above.
- Per-image exceptions are caught and logged (`ERROR {i}: {e}`) without stopping the run, but a full Blender crash (as above) still takes down the whole process, hence the recommendation to run outside the GUI.
