import bpy
import bmesh
import json
import math
import random
import os
import glob
import sys
import numpy as np
from mathutils import Vector, Matrix, Euler
from bpy_extras.object_utils import world_to_camera_view

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    "output_dir": r"D:\3drecognition\datasetCOCO4",
    "num_images": 35000,
    "start_index": 0,
    
    # === OCULUS QUEST BASE INTRINSICS ===
    "quest_fx_base": 866.68,
    "quest_fy_base": 866.68,
    "quest_cx_base": 638.57,
    "quest_cy_base": 480.49,
    "quest_width_base": 1280,
    "quest_height_base": 960,
    
    # === OUTPUT RESOLUTION ===
    "image_width": 1280,
    "image_height": 960,
    
    "use_quest_intrinsics": True,
    "scene_units":  "METERS",
    
    "background_images_dir": r"C:\Users\protd\Desktop\bg_images",
    "use_background_images": True,
    "background_image_prob": 0.95,
    
    # === DEGRADATION SETTINGS ===
    "degradation_prob": 0.5,
    
    # === DEPTH OF FIELD (OPTICAL) ===
    "enable_dof": False,
    "dof_fstop_range": (2.8, 8.0), 
    
    "camera_distance_range": (1.5, 5.0),
    "object_scale_range": (1.0, 1.0),
    "base_scale_for_export": 1.0, 
    "min_object_distance": 1.0,
    "rotation_range": {"x": (-180, 180), "y": (-180, 180), "z": (-180, 180)},
    
    "num_lights": 3,
    "light_energy_range": (70, 199),
    "cycles_samples": 32,
    "cleanup_interval": 50,
}


# ============================================
# POSE EXTRACTION & UTILS
# ============================================
def get_object_pose_in_camera_frame(obj, camera):
    obj_mat_world = obj.matrix_world
    loc, rot_quat, scale = obj_mat_world.decompose()
    rot_mat = rot_quat.to_matrix().to_4x4()
    loc_mat = Matrix.Translation(loc)
    obj_mat_no_scale = loc_mat @ rot_mat
    cam_mat = camera.matrix_world
    pose_mat = cam_mat.inverted() @ obj_mat_no_scale
    coord_transform = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
    pose_mat_cv = coord_transform @ pose_mat
    rotation = pose_mat_cv.to_3x3()
    translation = pose_mat_cv.translation
    R_flat = [float(rotation[r][c]) for r in range(3) for c in range(3)]
    T_mm = [float(t * 1000.0) for t in translation] if CONFIG["scene_units"] == "METERS" else [float(t) for t in translation]
    return {"R": R_flat, "T": T_mm}

def get_2d_bbox_coco(obj, camera, scene):
    coords_2d = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    render = scene.render
    width, height = render.resolution_x, render.resolution_y
    for v in mesh.vertices:
        world_v = obj.matrix_world @ v.co
        co_2d = world_to_camera_view(scene, camera, world_v)
        if co_2d.z > 0:
            coords_2d.append((co_2d.x * width, (1.0 - co_2d.y) * height))
    obj_eval.to_mesh_clear()
    if not coords_2d: return None
    min_x, max_x = max(0, min(c[0] for c in coords_2d)), min(width, max(c[0] for c in coords_2d))
    min_y, max_y = max(0, min(c[1] for c in coords_2d)), min(height, max(c[1] for c in coords_2d))
    w, h = max_x - min_x, max_y - min_y
    if w <= 0 or h <= 0: return None
    return {"x": float(min_x), "y": float(min_y), "width": float(w), "height": float(h), "area": float(w * h)}

def get_camera_intrinsics(camera, scene):
    if CONFIG.get("use_quest_intrinsics", False):
        scale_x = CONFIG["image_width"] / CONFIG["quest_width_base"]
        scale_y = CONFIG["image_height"] / CONFIG["quest_height_base"]
        return {
            "fx": float(CONFIG["quest_fx_base"] * scale_x),
            "fy": float(CONFIG["quest_fy_base"] * scale_y),
            "cx": float(CONFIG["quest_cx_base"] * scale_x),
            "cy": float(CONFIG["quest_cy_base"] * scale_y),
            "width": CONFIG["image_width"], "height": CONFIG["image_height"]
        }
    render = scene.render
    fx = (camera.data.lens / camera.data.sensor_width) * render.resolution_x
    return {
        "fx": float(fx), "fy": float(fx),
        "cx": float(render.resolution_x/2), "cy": float(render.resolution_y/2),
        "width": render.resolution_x, "height": render.resolution_y
    }

def export_models(target_objects, output_dir):
    models_dir = f"{output_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    models_info = {}
    base_scale = CONFIG.get("base_scale_for_export", 1.0)
    for idx, obj in enumerate(target_objects):
        orig_scale, orig_loc, orig_rot = obj.scale.copy(), obj.location.copy(), obj.rotation_euler.copy()
        obj.scale = (base_scale,)*3
        obj.location = (0,0,0)
        obj.rotation_euler = (0,0,0)
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.ply_export(filepath=f"{models_dir}/obj_{idx+1:06d}.ply", export_selected_objects=True)
        bbox = obj.bound_box
        min_v = [min(c[i] for c in bbox)*base_scale*1000 for i in range(3)]
        max_v = [max(c[i] for c in bbox)*base_scale*1000 for i in range(3)]
        size = [max_v[i]-min_v[i] for i in range(3)]
        models_info[str(idx+1)] = {
            "min_x": min_v[0], "min_y": min_v[1], "min_z": min_v[2],
            "size_x": size[0], "size_y": size[1], "size_z": size[2],
            "diameter": math.sqrt(sum(s**2 for s in size))
        }
        obj.scale, obj.location, obj.rotation_euler = orig_scale, orig_loc, orig_rot
    with open(f"{models_dir}/models_info.json", 'w') as f: json.dump(models_info, f, indent=2)

# ============================================
# COMPOSITOR & SCENE (BLUR & CHUNKY NOISE)
# ============================================
def setup_compositor_nodes(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    nodes, links = tree.nodes, tree.links
    nodes.clear()
    
    # 1. Input
    rl = nodes.new('CompositorNodeRLayers')
    rl.location = (0, 0)
    
    # 2. Blur Node (The "Soap Dish" effect)
    blur_node = nodes.new('CompositorNodeBlur')
    blur_node.filter_type = 'GAUSS'
    blur_node.size_x = 0 # Controlled in loop
    blur_node.size_y = 0
    blur_node.location = (200, 0)
    
    # 3. Brightness/Contrast
    contrast = nodes.new('CompositorNodeBrightContrast')
    contrast.location = (400, 0)
    
    # 4. Noise Texture
    tex_name = "ISONoiseTex"
    if tex_name not in bpy.data.textures:
        tex = bpy.data.textures.new(tex_name, 'CLOUDS')
        tex.noise_depth = 0
    else: tex = bpy.data.textures[tex_name]
    
    # We set a default here, but will change it in the loop for "Chunky" grain
    tex.noise_scale = 0.05 
    
    tex_node = nodes.new('CompositorNodeTexture')
    tex_node.texture = tex
    tex_node.location = (400, 200)
    
    # 5. Mix Node
    mix = nodes.new('CompositorNodeMixRGB')
    mix.blend_type = 'OVERLAY'
    mix.location = (600, 0)
    
    # 6. Output
    comp = nodes.new('CompositorNodeComposite')
    comp.location = (800, 0)
    
    # Pipeline: Render -> Blur -> Contrast -> Mix(Noise) -> Output
    links.new(rl.outputs[0], blur_node.inputs[0])
    links.new(blur_node.outputs[0], contrast.inputs[0])
    links.new(contrast.outputs[0], mix.inputs[1])
    links.new(tex_node.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], comp.inputs[0])
    
    return contrast, mix, blur_node

def setup_scene():
    """Initialize scene with proper settings and ROBUST GPU selection"""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if not obj.name.startswith("TargetObject") and obj.type != 'CAMERA':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    scene = bpy.context.scene
    scene.render.resolution_x = CONFIG["image_width"]
    scene.render.resolution_y = CONFIG["image_height"]
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    
    # === ROBUST GPU SETUP ===
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    
    # Try different backends in order of preference
    # OPTIX = Nvidia RTX cards (Best)
    # CUDA = Older Nvidia cards
    # HIP = AMD cards
    # METAL = Mac
    gpu_found = False
    
    for compute_type in ['OPTIX', 'CUDA', 'HIP', 'METAL']:
        try:
            cycles_prefs.compute_device_type = compute_type
            # CRITICAL: This refresh call is required in newer Blender versions
            cycles_prefs.get_devices()
            
            # Check if any devices of this type exist
            devices = [d for d in cycles_prefs.devices if d.type == compute_type]
            
            if devices:
                print(f"\n✅ GPU BACKEND FOUND: {compute_type}")
                for device in devices:
                    device.use = True
                    print(f"   - Enabled: {device.name}")
                gpu_found = True
                break # Stop searching if we found a valid backend
        except Exception as e:
            print(f"   Backend {compute_type} not available: {e}")
            continue
            
    if not gpu_found:
        print("\n⚠️ WARNING: No GPU devices found. Rendering will be slow (CPU)!")
        scene.cycles.device = 'CPU'
    # =========================
    
    scene.cycles.samples = CONFIG["cycles_samples"]
    scene.cycles.use_denoising = True
    scene.render.film_transparent = False
    scene.world.use_nodes = True

def create_camera():
    if bpy.context.scene.camera: cam = bpy.context.scene.camera
    else:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name = "MainCamera"
        bpy.context.scene.camera = cam
    
    if CONFIG.get("use_quest_intrinsics", False):
        fx = CONFIG["quest_fx_base"] * (CONFIG["image_width"] / CONFIG["quest_width_base"])
        cam.data.sensor_fit = 'HORIZONTAL'
        cam.data.sensor_width = 36.0
        cam.data.lens = (fx * 36.0) / CONFIG["image_width"]
    else:
        cam.data.lens_unit = 'FOV'
        cam.data.angle = math.radians(67)
    
    if CONFIG["enable_dof"]:
        cam.data.dof.use_dof = True
    
    return cam

def create_lights():
    for i in range(CONFIG["num_lights"]):
        bpy.ops.object.light_add(type='POINT')
        light = bpy.context.object
        light.location = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(3, 7))
        light.data.energy = random.uniform(*CONFIG["light_energy_range"])

def setup_background_color():
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    out = nodes.new('ShaderNodeOutputWorld')
    bg.inputs[0].default_value = (random.random(), random.random(), random.random(), 1.0)
    bg.inputs[1].default_value = random.uniform(0.1, 0.3)
    world.node_tree.links.new(bg.outputs[0], out.inputs[0])

def setup_background_image(image_files):
    if not image_files: return setup_background_color()
    img_path = random.choice(image_files)
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    nodes.clear()
    tex = nodes.new('ShaderNodeTexEnvironment')
    bg = nodes.new('ShaderNodeBackground')
    out = nodes.new('ShaderNodeOutputWorld')
    try:
        img_name = os.path.basename(img_path)
        if img_name in bpy.data.images: image = bpy.data.images[img_name]
        else: image = bpy.data.images.load(img_path)
        tex.image = image
    except Exception: return setup_background_color()
    
    bg.inputs[1].default_value = random.uniform(0.1, 0.3)
    world.node_tree.links.new(tex.outputs['Color'], bg.inputs['Color'])
    world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])
    tex.texture_mapping.rotation[2] = random.uniform(0, 6.28)

# ============================================
# MAIN GENERATION LOOP
# ============================================
def check_collision(new_loc, existing_locs, min_dist):
    for loc in existing_locs:
        if (new_loc - loc).length < min_dist: return True
    return False

def randomize_object_pose(obj, existing_locations=[]):
    obj.rotation_euler = Euler((random.uniform(*CONFIG["rotation_range"]["x"])*math.pi/180, random.uniform(*CONFIG["rotation_range"]["y"])*math.pi/180, random.uniform(*CONFIG["rotation_range"]["z"])*math.pi/180), 'XYZ')
    scale = random.uniform(*CONFIG["object_scale_range"])
    obj.scale = (scale, scale, scale)
    for _ in range(50):
        loc = Vector((random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), random.uniform(-0.1, 0.1)))
        if not check_collision(loc, existing_locations, CONFIG["min_object_distance"]):
            obj.location = loc
            return loc
    obj.location = Vector((0, 0, 0))
    return Vector((0, 0, 0))

def randomize_camera_pose(camera):
    dist = random.uniform(*CONFIG["camera_distance_range"])
    theta, phi = random.uniform(0, 2*math.pi), random.uniform(math.pi/6, math.pi/2.5)
    loc = Vector((dist*math.sin(phi)*math.cos(theta), dist*math.sin(phi)*math.sin(theta), dist*math.cos(phi)))
    camera.location = loc
    camera.rotation_euler = (Vector((0,0,0))-loc).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

def cleanup_memory():
    for block in bpy.data.meshes: 
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.images:
        if block.users == 0 and block.name != 'Render Result': bpy.data.images.remove(block)
    import gc; gc.collect()

def generate_dataset():
    output_dir = CONFIG['output_dir']
    os.makedirs(f"{output_dir}/annotations", exist_ok=True)
    os.makedirs(f"{output_dir}/train_pbr/000000/rgb", exist_ok=True)
    
    bg_files = []
    if CONFIG["use_background_images"] and os.path.exists(CONFIG["background_images_dir"]):
        bg_files = glob.glob(os.path.join(CONFIG["background_images_dir"], "*.*"))
    if len(bg_files) > 100: bg_files = random.sample(bg_files, 100)
    
    setup_scene()
    
    # Get nodes including Blur node
    contrast_node, mix_node, blur_node = setup_compositor_nodes(bpy.context.scene)
    
    camera = create_camera()
    target_objects = sorted([o for o in bpy.data.objects if o.name.startswith("TargetObject")], key=lambda x: x.name)
    if not target_objects: return print("ERROR: No TargetObject found!")
        
    export_models(target_objects, output_dir)
    
    coco_dataset = {"images": [], "annotations": [], "categories": []}
    for idx, obj in enumerate(target_objects):
        coco_dataset["categories"].append({"id": idx, "name": obj.name, "supercategory": "object"})
    
    with open(f"{output_dir}/camera.json", 'w') as f: json.dump(get_camera_intrinsics(camera, bpy.context.scene), f, indent=2)
    
    annotation_id = 0
    print(f"Generating {CONFIG['num_images']} images...")
    
    for i in range(CONFIG["start_index"], CONFIG["start_index"] + CONFIG["num_images"]):
        try:
            if bg_files and random.random() < CONFIG["background_image_prob"]: setup_background_image(bg_files)
            else: setup_background_color()
            
            if i % 10 == 0:
                for o in bpy.data.objects: 
                    if o.type == 'LIGHT': bpy.data.objects.remove(o)
                create_lights()
            
            # === DEGRADATION LOGIC (Soap Dish / Bad Camera Effect) ===
            tex = bpy.data.textures["ISONoiseTex"]
            
            if random.random() < CONFIG["degradation_prob"]:
                # 1. Darker / Low Contrast
                contrast_node.inputs[2].default_value = random.uniform(-0.9, -0.6) # Contrast
                contrast_node.inputs[1].default_value = random.uniform(-0.3, -0.1) # Brightness
                
                # 2. Stronger Noise (Mix Factor)
                mix_node.inputs[0].default_value = random.uniform(0.4, 0.8)
                
                # 3. BIGGER GRAIN (The requested size increase)
                # 0.1 to 0.3 makes it look like chunky digital sensor noise
                tex.noise_scale = random.uniform(0.005, 0.009)
                
                # 4. BLUR (Bad Lens / Soap Dish Effect)
                # Random blur between 2 and 6 pixels radius
                blur_val = random.randint(0, 6)
                blur_node.size_x = blur_val
                blur_node.size_y = blur_val
                
            else:
                # Clean Image
                contrast_node.inputs[2].default_value = 0.0
                contrast_node.inputs[1].default_value = 0.0
                mix_node.inputs[0].default_value = 0.0
                tex.noise_scale = 0.005 # Reset to tiny grain
                blur_node.size_x = 0
                blur_node.size_y = 0
            
            randomize_camera_pose(camera)
            
            # Random F-Stop for background bokeh (Depth of Field)
            if CONFIG["enable_dof"]:
                camera.data.dof.aperture_fstop = random.uniform(*CONFIG["dof_fstop_range"])
            
            for o in target_objects: 
                o.hide_render = True
                o.location = (100, 100, 100)
            
            selected = random.sample(target_objects, random.randint(1, len(target_objects)))
            locs, image_annotations = [], []
            
            # Set Focus Object for Depth of Field
            if CONFIG["enable_dof"] and selected:
                camera.data.dof.focus_object = selected[0]
            
            for idx, obj in enumerate(target_objects):
                if obj in selected:
                    obj.hide_render = False
                    locs.append(randomize_object_pose(obj, locs))
                    bpy.context.view_layer.update()
                    pose = get_object_pose_in_camera_frame(obj, camera)
                    bbox = get_2d_bbox_coco(obj, camera, bpy.context.scene)
                    if bbox:
                        image_annotations.append({
                            "id": annotation_id, "image_id": i, "category_id": idx,
                            "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']], "area": bbox['area'],
                            "iscrowd": 0, "R": pose['R'], "T": pose['T']
                        })
                        annotation_id += 1
            
            if image_annotations:
                fname = f"{i:06d}.png"
                bpy.context.scene.render.filepath = f"{output_dir}/train_pbr/000000/rgb/{fname}"
                bpy.ops.render.render(write_still=True)
                coco_dataset["images"].append({
                    "id": i, "file_name": f"train_pbr/000000/rgb/{fname}",
                    "width": CONFIG["image_width"], "height": CONFIG["image_height"],
                    "image_folder": "000000", "type": "pbr"
                })
                coco_dataset["annotations"].extend(image_annotations)
            
            if (i + 1) % CONFIG["cleanup_interval"] == 0:
                cleanup_memory()
                with open(f"{output_dir}/annotations/instances_train.json", 'w') as f: json.dump(coco_dataset, f)
                print(f"Generated {i + 1}/{CONFIG['start_index'] + CONFIG['num_images']}")
                
        except Exception as e: print(f"ERROR {i}: {e}")
            
    with open(f"{output_dir}/annotations/instances_train.json", 'w') as f: json.dump(coco_dataset, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    generate_dataset()