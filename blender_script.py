import argparse
import sys
import os
import math
import glob
from typing import Dict, Callable, Tuple, Literal
import json

import numpy as np

import bpy
import bmesh
import mathutils
from mathutils import Vector


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False):
    if not any([save_depth, save_normal, save_albedo, save_mist]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['View Layer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['View Layer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['View Layer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['View Layer'].use_pass_mist = save_mist
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
        
    return outputs, spec_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera(arg):
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Set the camera to orthographic mode and adjust its scale
    if not arg.perspective_camera:
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 1.75
    else:
        cam.data.type = 'PERSP'

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    cam_empty.name = "Camera_Target"
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)


def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['View Layer'].material_override = new_mat

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def get_self_and_children(object): 
    children = [object] 
    for ob in object.children:
        children.extend(get_self_and_children(ob))
    return children 

def object_bbox(object):
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_self_and_children(object):
        if obj.type != "MESH":
            continue

        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def load_and_place_tiles(tile_descriptor, axes, debase=False):
    movement_axis_x, movement_axis_y = axes

    already_loaded = [obj for obj in bpy.context.scene.objects.values() if obj.parent is None]
    tiles = []

    for tile_dict in tile_descriptor:
        load_object(tile_dict["path"])
        
        scene = bpy.data.objects.new("Tile", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in [obj for obj in bpy.context.scene.objects.values() if obj.parent is None and obj not in already_loaded and obj is not scene]:
            obj.parent = scene
        
        for obj in [obj for obj in bpy.context.scene.objects.values() if obj.parent is None and obj not in already_loaded]:
            obj["max_corner"] = get_top_margin_z(obj)
            obj["position"] = Vector((tile_dict["x"], tile_dict["y"]))
            obj["has_slab"] = tile_dict["has_slab"] if "has_slab" in tile_dict else True

            if "slice_z" in tile_dict:
                obj["slice_z"] = tile_dict["slice_z"]

            if debase:
                obj["plane_cos"], obj["plane_nos"] = debase_tile(obj, tile_dict["rgb_cut"] if "rgb_cut" in tile_dict else None)

            obj["scale_factor"], obj["translation"] = scale_object_to_unit(obj)

            # we might need to re-translate the object to the origin
            # since the debase_tile function could in rare cases lead to
            # a translation of the object
            debase_bbox_min, debase_bbox_max = object_bbox(obj)
            debase_bbox_extent = debase_bbox_max - debase_bbox_min

            # calculate offset to (0, 0)
            debase_offset = (debase_bbox_min + debase_bbox_extent / 2) * Vector((1, 1, 0))

            T_offset = mathutils.Matrix.Translation(-debase_offset)
            for lobj in get_self_and_children(obj):
                lobj.matrix_world = T_offset @ lobj.matrix_world

            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

            if debase_offset is not None:
                obj["translation"] = Vector(obj["translation"]) + debase_offset

            tile_min, tile_max = object_bbox(obj)
            tile_extent = tile_max - tile_min

            T_offset = mathutils.Matrix.Translation(
                tile_dict["x"] * movement_axis_x * tile_extent +
                tile_dict["y"] * movement_axis_y * tile_extent
            )

            bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

            if "rotation" in tile_dict:
                R = mathutils.Matrix.Rotation(np.radians(tile_dict["rotation"]), 4, 'Z')
            else:
                R = mathutils.Matrix.Identity(4)

            for lobj in get_self_and_children(obj):
                lobj.matrix_world = T_offset @ R @ lobj.matrix_world
            
            already_loaded.append(obj)
            tiles.append(obj)
            
    return tiles

def create_slab(location, size=1.0, height=0.1, z_offset=0.15, topless=False):
    """
    Creates a xz unit-sized slab with a certain z height and an offset to the (z) bottom of the unit cube.
    """
    
    assert location.z == 0
    
    # Create a plane and immediately switch to edit mode
    #bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=True, location=(0, 0, 0.5))#(0, 0, -(0.5-z_offset-height)))
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=True, location=(location.x, location.y, -(0.5-z_offset-height)))
    bpy.context.active_object.name = "Slab"

    # Get the active mesh in edit mode
    bm = bmesh.from_edit_mesh(bpy.context.active_object.data)

    # Ensure we have all faces selected (by default the new plane will be selected)
    for face in bm.faces:
        face.select = True

    # Extrude the selected face downward along the z-axis (e.g. by -0.1 units)
    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, -height)}
        #TRANSFORM_OT_translate={"value": (0, 0, -1)}
    )

    if topless:
        # go through all faces and delete the top face
        max_z = -math.inf
        max_face = None
        for face in bm.faces:
            for vert in face.verts:
                if vert.co.z > max_z:
                    max_z = vert.co.z
                    max_face = face

        bm.faces.remove(max_face)

    # Go back to object mode to finish
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return bpy.context.active_object

def compute_uv_area(poly, uv_layer):
    """Compute the area in UV space for a polygon using the shoelace formula."""
    uv_coords = [uv_layer.data[i].uv for i in poly.loop_indices]
    area = 0.0
    n = len(uv_coords)
    for i in range(n):
        x1, y1 = uv_coords[i]
        x2, y2 = uv_coords[(i+1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0

def compute_average_uv_texture_color(source_obj, min_z):
    """
    Computes an average color from the UV texture of source_obj.
    For each face, the UV centroid is computed and then used to sample
    the texture image. Each face's contribution is weighted by its UV area.
    """
    mesh = source_obj.data
    uv_layer = mesh.uv_layers.active
    if not uv_layer:
        return (0.8, 0.8, 0.8, 1.0)  # Fallback color if no UVs
    
    # Look for an image texture node in the first material.
    mat = source_obj.data.materials[0] if source_obj.data.materials else None
    image = None
    if mat and mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                image = node.image
                break
    if not image:
        return (0.8, 0.8, 0.8, 1.0)  # Fallback if no image found
    
    # Get the image's dimensions and pixel data.
    width, height = image.size[0], image.size[1]
    # The pixels come as a flat list of floats (RGBA repeated).
    pixels = np.array(image.pixels[:])
    pixels = pixels.reshape(-1, 4)
    
    total_weight = 0.0
    weighted_color = np.array([0.0, 0.0, 0.0, 0.0])
    
    for poly in mesh.polygons:
        poly_center = poly.center
        # convert to world coordinates
        poly_center_world = source_obj.matrix_world @ poly_center
        if poly_center_world.z < min_z:
            continue

        # Compute the UV centroid of the polygon.
        uv_centroid = np.array([0.0, 0.0])
        for li in poly.loop_indices:
            uv = uv_layer.data[li].uv
            uv_centroid += np.array(uv)
        uv_centroid /= len(poly.loop_indices)
        
        # Compute the weight from the polygon's UV area.
        weight = compute_uv_area(poly, uv_layer)
        
        # Convert the UV coordinate to pixel coordinates.
        u, v = uv_centroid
        x = int(u * (width - 1))
        y = int(v * (height - 1))
        idx = (y * width + x)
        face_color = pixels[idx]  # This is an array like [r, g, b, a]
        
        weighted_color += np.array(face_color) * weight
        total_weight += weight
    
    if total_weight > 0:
        avg = weighted_color / total_weight
    else:
        avg = np.array([0.8, 0.8, 0.8, 1.0])

    # Force full opacity in the result.
    return (avg[0], avg[1], avg[2], 1.0)
    
def create_average_color_holdout_surface(location, size=1.0, min_z=0.0, source_obj=None):
    """
    Creates a plane at the given location with the given size,
    positioned at z=min_z, and assigns it a material whose diffuse
    color is computed as the average color of all faces of source_obj.
    Assumes source_obj's material is UV texture based.
    
    Parameters:
        location: A mathutils.Vector representing the x,y location.
        size: The size of the plane.
        min_z: The z coordinate where the plane should be placed.
        source_obj: The object whose material's average color will be computed.
        
    Returns:
        The created holdout surface object.
    """

    # Default average color if no source is provided.
    avg_color = (0.8, 0.8, 0.8, 1.0)
    if source_obj is not None:
        avg_color = compute_average_uv_texture_color(source_obj, min_z)
    
    # Create a new material that uses the computed average color.
    new_mat = bpy.data.materials.new(name="AvgColorHoldoutMat")
    new_mat.use_nodes = True
    nodes = new_mat.node_tree.nodes
    links = new_mat.node_tree.links

    # Clear default nodes.
    for node in nodes:
        nodes.remove(node)

    # Create a Diffuse BSDF node.
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = avg_color
    diffuse.location = (0, 0)

    # Create an Output node.
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (200, 0)
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])

    # Create the holdout surface plane.
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, min_z))
    holdout_plane = bpy.context.active_object
    holdout_plane.name = "AvgColorHoldoutSurface"

    # Assign the new material to the holdout plane.
    if len(holdout_plane.data.materials) == 0:
        holdout_plane.data.materials.append(new_mat)
    else:
        holdout_plane.data.materials[0] = new_mat
        
    T_offset = mathutils.Matrix.Translation(location)

    for lobj in get_self_and_children(holdout_plane):
        lobj.matrix_world = T_offset @ lobj.matrix_world

    return holdout_plane

def create_masked_cube(location, size=1.0, height=0.85-0.1, z_offset=0.15+0.1):
    assert location.z == 0

    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=True, location=(location.x, location.y, -(0.5-z_offset-height)))
    bpy.context.active_object.name = "Slab"

    bm = bmesh.from_edit_mesh(bpy.context.active_object.data)

    for face in bm.faces:
        face.select = True

    bpy.ops.mesh.extrude_region_move(
        TRANSFORM_OT_translate={"value": (0, 0, -height)}
    )

    bpy.ops.object.mode_set(mode='OBJECT')
    
    cube = bpy.context.active_object

    mat = bpy.data.materials.new(name="PerfectWhite")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    for node in nodes:
        nodes.remove(node)

    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission_node.inputs["Strength"].default_value = 100.0
    emission_node.location = (0, 0)

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (200, 0)

    links = mat.node_tree.links
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    cube.data.materials.append(mat)
    
    return cube

def create_holdout_black_material():
    mat = bpy.data.materials.new(name="HoldoutBlack")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    holdout_node = nodes.new(type='ShaderNodeHoldout')
    holdout_node.location = (0, 0)

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (200, 0)

    links.new(holdout_node.outputs["Holdout"], output_node.inputs["Surface"])

    return mat

def set_holdout_and_background(mode: Literal["transparent", "black"]):
    if mode != "transparent":
        world = bpy.data.worlds["World"]
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # Black RGBA

    scene = bpy.context.scene
    scene.render.film_transparent = (mode == "transparent")  # This ensures holdout areas are transparent

    scene.use_nodes = True
    tree = scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    rlayers = tree.nodes.new(type='CompositorNodeRLayers')
    rlayers.location = (0, 0)

    if mode != "transparent":
        bg_node = tree.nodes.new(type='CompositorNodeRGB')
        bg_node.outputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # Black
        bg_node.location = (0, -200)

        alpha_over = tree.nodes.new(type='CompositorNodeAlphaOver')
        alpha_over.location = (200, 0)

    composite = tree.nodes.new(type='CompositorNodeComposite')
    composite.location = (400, 0)

    if mode != "transparent":
        tree.links.new(rlayers.outputs["Image"], alpha_over.inputs[2])
        tree.links.new(bg_node.outputs[0], alpha_over.inputs[1])
        tree.links.new(alpha_over.outputs["Image"], composite.inputs["Image"])
    
    else:
        tree.links.new(rlayers.outputs["Image"], composite.inputs["Image"])
    
def get_margin_vertices(obj, tol=1e-3):
    """Return all vertices (in local coordinates) that lie near the object's margin.
    We define margin vertices as those whose x or y coordinate is within tol of
    the bounding box minimum or maximum.
    """
    if obj.type != "MESH":
        obj = [m for m in get_self_and_children(obj) if m.type=="MESH"][0]

    bbox_min, bbox_max = object_bbox(obj)
    candidates = []
    for v in obj.data.vertices:
        co = v.co
        if (abs(co.x - bbox_min.x) < tol or abs(co.x - bbox_max.x) < tol or
            abs(co.y - bbox_min.y) < tol or abs(co.y - bbox_max.y) < tol):
            candidates.append(co)
    return candidates

def get_top_margin_z(obj, tol=1e-3, percentile=90, convert_to_world=True):
    """
    Compute a representative z value for the top surface of the base.
    It does so by filtering margin vertices to only consider those in the
    top percentile of z values, thus ignoring vertices from the bottom of the base.
    """
    margin_verts = get_margin_vertices(obj, tol)

    if convert_to_world:
        margin_verts = [obj.matrix_world @ co for co in margin_verts]
    
    coords = np.array([[v.x, v.y, v.z] for v in margin_verts])
    z_vals = coords[:, 2]
    
    z_threshold = np.percentile(z_vals, percentile)
    
    top_verts = [v for v in margin_verts if v.z >= z_threshold]
    
    top_z = np.median([v.z for v in top_verts])

    return top_z

def bisect(obj, plane_cos, plane_nos, world_cut=False):
    if obj.type != "MESH":
        obj_mesh = [m for m in get_self_and_children(obj) if m.type=="MESH"][0]
    else:
        obj_mesh = obj

    bpy.context.view_layer.objects.active = obj_mesh
    bpy.ops.object.mode_set(mode='EDIT')

    if not world_cut:
        bm = bmesh.from_edit_mesh(obj_mesh.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        
        for (plane_co, plane_no) in zip(plane_cos, plane_nos):
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(
                bm,
                geom=geom,
                plane_co=plane_co,
                plane_no=plane_no,
                clear_outer=True,   # This deletes geometry in the direction of the normal (i.e. bottom part).
                clear_inner=False,
                use_snap_center=False
            )
            bmesh.update_edit_mesh(obj_mesh.data)
        
        bmesh.update_edit_mesh(obj_mesh.data)
        bpy.ops.object.mode_set(mode='OBJECT')

    else:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        obj_mesh.select_set(True)
        bpy.context.view_layer.objects.active = obj_mesh
        bpy.ops.object.mode_set(mode='EDIT')

        for plane_co, plane_no in zip(plane_cos, plane_nos):
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.bisect(
                plane_co=(plane_co.x, plane_co.y, plane_co.z),
                plane_no=(plane_no.x, plane_no.y, plane_no.z),
                clear_outer=True,   # Deletes geometry on the side of the normal.
                clear_inner=False,
                use_fill=False
            )

    bpy.ops.object.mode_set(mode='OBJECT')
    
def debase_tile(obj, cut_dict=None, cut_padding=3.5e-2):
    tile_min, tile_max = object_bbox(obj)
    tile_extent = tile_max - tile_min

    # for tiles that were generated not perfectly square, we need to make
    # them perfectly square. otherwise, lots of assumptions won't work
    assumed_tile_extent = min(tile_extent.x, tile_extent.y)

    if cut_dict is None:
        plane_cos = [Vector(v) for v in
            (
                (obj.location.x + (assumed_tile_extent * .5 - cut_padding), 0, 0),
                (obj.location.x - (assumed_tile_extent * .5 - cut_padding), 0, 0),
                (0, obj.location.y + (assumed_tile_extent * .5 - cut_padding), 0),
                (0, obj.location.y - (assumed_tile_extent * .5 - cut_padding), 0),
            )
        ]
    else:
        plane_cos = [Vector(v) for v in
            (
                (cut_dict["max_x"], 0, 0),
                (cut_dict["min_x"], 0, 0),
                (0, cut_dict["max_y"], 0),
                (0, cut_dict["min_y"], 0),
            )
        ]

    plane_nos = [Vector(v) for v in
        (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
        )
    ]
    
    bisect(obj, plane_cos, plane_nos)

    return plane_cos, plane_nos
    
def apply_global_cut(cut_at, axes, size=1., slice_bottom=False):
    bpy.ops.object.select_all(action='DESELECT')

    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    for obj in mesh_objs:
        obj.select_set(True)
    
    bpy.context.view_layer.objects.active = mesh_objs[0]
    
    bpy.ops.object.join()
        
    obj = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH'][0]

    T_offset = mathutils.Matrix.Translation(-(cut_at.x * axes[0] + cut_at.y * axes[1]))

    for lobj in get_self_and_children(obj):
        lobj.matrix_world = T_offset @ lobj.matrix_world
        
    tile_min, _ = object_bbox(obj)

    plane_cos = [Vector(v) for v in [
        (size/2, 0, 0),
        (-size/2, 0, 0),
        (0, size/2, 0),
        (0, -size/2, 0),
    ]]

    plane_nos = [Vector(v) for v in [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    ]]

    if slice_bottom:
        plane_cos.append(Vector((0, 0, tile_min.z + 1e-2)))
        plane_nos.append(Vector((0, 0, -1)))

    bisect(obj, plane_cos, plane_nos, world_cut=True)

def attach_slab_to_tile(slab, tile, k=4):
    if tile.type != "MESH":
        tile = [m for m in get_self_and_children(tile) if m.type=="MESH"][0]
        
    if slab.type != "MESH":
        slab = [m for m in get_self_and_children(slab) if m.type=="MESH"][0]

    verts_tile_world = [tile.matrix_world @ v.co for v in tile.data.vertices]

    bpy.ops.object.mode_set(mode='OBJECT')

    vertices_plus_dist = []
    for i, v in enumerate(slab.data.vertices):
        v_world = slab.matrix_world @ v.co

        min_dist = float('inf')
        closest_world = None
        for a in verts_tile_world:
            d = (v_world - a).length
            if d < min_dist:
                min_dist = d
                closest_world = a
        vertices_plus_dist.append((i, min_dist, closest_world))

    sorted_dist = sorted(vertices_plus_dist, key=lambda x: x[1])

    if k > 0:
        sorted_dist = sorted_dist[:k]
        
    for i, _, closest_world in sorted_dist:
        if closest_world is not None:
            new_local = slab.matrix_world.inverted() @ closest_world
            slab.data.vertices[i].co = new_local

    slab.data.update()

def is_adjacent(tile1, tile2):
    return np.isclose((Vector(tile1["position"])-Vector(tile2["position"])).length, 1.)

def scale_object_to_unit(obj):
    bbox_min, bbox_max = object_bbox(obj)
    
    extent = bbox_max - bbox_min
    
    side_length = extent.x
    if side_length == 0:
        return
    
    scale_factor = 1.0 / side_length
    
    obj.scale = obj.scale * scale_factor
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    diff = Vector((0, 0, obj["max_corner"] * (1 - scale_factor)))
    
    T_offset = mathutils.Matrix.Translation(diff)

    for lobj in get_self_and_children(obj):
        lobj.matrix_world = T_offset @ lobj.matrix_world
        
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    return scale_factor, diff

def four_cut_bisect(obj, obj_pos, z, z2):
    """
    Performs four bisect cuts on obj to remove everything above the sloping surfaces
    defined by lines running from (0, 0, z) to the object's bounding box edges at z2.
    
    The cuts are:
      - Left:  from (0,0,z) to (bbox_min.x, 0, z2)
      - Right: from (0,0,z) to (bbox_max.x, 0, z2)
      - Bottom: from (0,0,z) to (0, bbox_min.y, z2)
      - Top:   from (0,0,z) to (0, bbox_max.y, z2)
    
    The plane normals are computed so that using clear_outer=True with the bisect operator
    will delete all geometry “above” the inclined plane.
    
    Assumes:
      - z2 is smaller than z.
      - The object's bounding box (in its local space) gives valid bbox_min and bbox_max.
    """
    
    # Get the object's bounding box in local coordinates.
    bbox_min, bbox_max = object_bbox(obj)
    
    # LEFT CUT: from (0,0,z) to (bbox_min.x, 0, z2)
    # Slope m = (z2 - z) / (bbox_min.x - 0). (bbox_min.x is negative so m > 0)
    left_m = (z2 - z) / bbox_min.x
    left_plane_no = Vector((-left_m, 0, 1)).normalized()
    
    # RIGHT CUT: from (0,0,z) to (bbox_max.x, 0, z2)
    # Here, bbox_max.x is positive so (z2 - z)/bbox_max.x is negative.
    right_m = (z2 - z) / bbox_max.x
    right_plane_no = Vector((-right_m, 0, 1)).normalized()
    
    # BOTTOM CUT: from (0,0,z) to (0, bbox_min.y, z2)
    bottom_m = (z2 - z) / bbox_min.y  # bbox_min.y is negative so bottom_m > 0
    bottom_plane_no = Vector((0, -bottom_m, 1)).normalized()
    
    # TOP CUT: from (0,0,z) to (0, bbox_max.y, z2)
    top_m = (z2 - z) / bbox_max.y  # bbox_max.y is positive so top_m is negative
    top_plane_no = Vector((0, -top_m, 1)).normalized()
    
    # All planes share the same origin at (0, 0, z)
    plane_co = Vector((obj_pos.x, obj_pos.y, z))
    plane_cos = [plane_co, plane_co, plane_co, plane_co]
    plane_nos = [left_plane_no, right_plane_no, bottom_plane_no, top_plane_no]
    
    # Now call your bisect function using world_cut=True.
    bisect(obj, plane_cos, plane_nos, world_cut=True)

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist
    )

    init_scene()
    
    tile_descriptor = json.loads(arg.tiles) if arg.tiles is not None else []
    
    movement_axis_x, movement_axis_y = Vector((1, 0, 0)), Vector((0, 1, 0))

    tiles = load_and_place_tiles(tile_descriptor, axes=(movement_axis_x, movement_axis_y), debase=arg.debase)

    # we set a fixed offset to the bottom. this allows us to maintain a consistent inpainting mask
    z_offset = 0.15
    slab_height = 0.1

    if len(tiles) > 0 and not arg.no_tile_modification:
        # align the height of all tiles based on z_offset
        for tile in tiles:
            delta = (-(0.5-z_offset) + slab_height) - tile["max_corner"]

            # move the entire world so that the cut is at the origin
            T_offset = mathutils.Matrix.Translation(Vector((0, 0, delta)))

            if "translation" in tile:
                tile["translation"] = Vector(tile["translation"]) + Vector((0, 0, delta))
            else:
                tile["translation"] = Vector((0, 0, delta))

            for lobj in get_self_and_children(tile):
                lobj.matrix_world = T_offset @ lobj.matrix_world
    
    holdout_material = create_holdout_black_material()

    set_holdout_and_background("transparent")

    has_next_slab_tile = False
    if not arg.no_slabs and arg.next_tile_at is not None:
        next_tile_at = Vector([float(c) for c in arg.next_tile_at.split(",")])
        slab = create_slab(Vector((next_tile_at.x, next_tile_at.y, 0)), height=slab_height, z_offset=z_offset)
        has_next_slab_tile = True

    additional_slabs = []
    if len(tiles) > 0 and not arg.no_slabs:
        for i in range(len(tiles)):
            if not tiles[i]["has_slab"]:
                continue

            tile_slab = create_slab(
                tile_descriptor[i]["x"] * movement_axis_x +
                tile_descriptor[i]["y"] * movement_axis_y, height=slab_height, z_offset=z_offset, topless=True, size=1+1e-3)
            
            tiles[i]["slab"] = tile_slab
            
            additional_slabs.append(tile_slab)
      
    if not arg.no_slabs:
        for base_tile in tiles:
            for tile in [t for t in tiles if t != base_tile and is_adjacent(t, base_tile)]:
                if "slab" in base_tile and "slab" in tile:
                    attach_slab_to_tile(base_tile["slab"], tile["slab"], k=4)

    for tile in tiles:
        # slice z if present
        if "slice_z" in tile:
            tile_pos = tile["position"][0] * movement_axis_x + tile["position"][1] * movement_axis_y
            four_cut_bisect(tile, tile_pos, float(tile["slice_z"]), float(tile["max_corner"] + tile["translation"][-1]))
            
            if not arg.no_slabs:
                tile_min, tile_max = object_bbox(tile)
                tile_extent = tile_max - tile_min
                surface_position = tile["position"][0] * movement_axis_x + tile["position"][1] * movement_axis_y
                tile["surface"] = create_average_color_holdout_surface(surface_position, size=min(tile_extent.x, tile_extent.y), min_z=-0.5+z_offset+0.75*slab_height, source_obj=[m for m in get_self_and_children(tile) if m.type=="MESH"][0])

        # and slice too big pots of the tiles at the bottom
        if tile["has_slab"]:
            bisect(tile, [Vector((0, 0, -0.5+z_offset+1e-3))], [Vector((0, 0, -1))], world_cut=True)
        elif not arg.no_tile_modification:
            # pretty aggressive cut here to prevent a "floating" bottom base
            bisect(tile, [Vector((0, 0, -0.5+z_offset+slab_height*0.5))], [Vector((0, 0, -1))], world_cut=True)
            
    if arg.cut_at is not None:
        assert len(arg.cut_at.split(',')) >= 2
        assert arg.rgb_only
        
        cut_coordinates = [float(c) for c in arg.cut_at.split(',')]
        apply_global_cut(Vector(cut_coordinates), axes=(movement_axis_x, movement_axis_y), size=1., slice_bottom=True)
        
    elif arg.next_tile_at is not None:
        T_offset = mathutils.Matrix.Translation(-Vector((next_tile_at.x, next_tile_at.y, 0)))

        to_move = tiles
        if not arg.no_slabs:
            to_move += additional_slabs
            
            for tile in tiles:
                if "surface" in tile:
                    to_move.append(tile["surface"])

            if has_next_slab_tile:
                to_move.append(slab)

        for obj in to_move:
            for lobj in get_self_and_children(obj):
                lobj.matrix_world = T_offset @ lobj.matrix_world

    if arg.export_tile_info:
        tile_info = []

        for tile in tiles:
            position = {k: int(v) for (k, v) in zip(("x", "y"), tile["position"].to_list())}

            tile_info.append({
                **position,
                "max_corner": tile["max_corner"],
                "scale_factor": tile["scale_factor"],
                "translation": tile["translation"].to_list(),
                **{k: [x.to_list() for x in tile[k]] for k in ("plane_cos", "plane_nos")},
            })

        with open(os.path.join(arg.output_folder, "tile_info.json"), "w") as f:
            json.dump(tile_info, f)

    if arg.save_mesh:        
        bpy.ops.wm.obj_export(filepath=os.path.join(arg.output_folder, 'export.obj'), export_materials=False)

    if arg.no_render:
        return
    
    # Initialize camera and lighting
    cam = init_camera(arg)

    if not arg.no_lights:
        init_lighting()
    else:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        if bpy.context.scene.world and bpy.context.scene.world.use_nodes:
            # Assumes a node named "Background" is present.
            bg = bpy.context.scene.world.node_tree.nodes.get("Background")
            if bg:
                bg.inputs[1].default_value = 0.0  # Set strength to 0

        # Backup dictionary for restoring original node trees later (if desired).
        original_material_trees = {}

        # Iterate over all materials in the blend file.
        for mat in bpy.data.materials:
            if not mat.use_nodes:
                continue  # Skip materials that don't use nodes.
            
            # Backup the original node tree.
            original_material_trees[mat.name] = mat.node_tree.copy()
            
            # Get the node tree and clear it.
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            nodes.clear()
            
            # Try to find an image texture node in the original node tree.
            image = None
            for node in original_material_trees[mat.name].nodes:
                if node.type == 'TEX_IMAGE' and node.image is not None:
                    image = node.image
                    break

            # If an image was found, create a new Image Texture node.
            if image:
                tex_node = nodes.new('ShaderNodeTexImage')
                tex_node.image = image
                tex_node.location = (-300, 0)
            
            # Create an Emission node.
            emission_node = nodes.new('ShaderNodeEmission')
            emission_node.location = (0, 0)
            
            # Connect the texture node's colour to the emission shader, if available.
            if image:
                links.new(tex_node.outputs['Color'], emission_node.inputs['Color'])
            else:
                # If no texture is found, use a default white colour.
                emission_node.inputs['Color'].default_value = (1, 1, 1, 1)
            
            # Create the Material Output node.
            output_node = nodes.new('ShaderNodeOutputMaterial')
            output_node.location = (300, 0)
            links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    print('[INFO] Camera and lighting initialized.')

    # Override material
    if arg.geo_mode:
        override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "frames": []
    }

    if arg.views is None:
        views = [{"yaw": np.radians(-45), "pitch": np.arctan(1/np.sqrt(2)), "radius": 2, "fov": np.radians(47.1)}]
    else:
        views = json.loads(arg.views)
        
    if arg.rgb_only:
        modes = ("rgb",)
    else:
        modes = ("rgb", "inpaint_mask", "conditioning_mask")

    for mode in modes:
        for i, view in enumerate(views):
            # reset to user preferences
            if arg.skip_rendering_tiles:
                for tile in tiles:
                    for obj in get_self_and_children(tile):
                        obj.hide_render = True
                        
                    if "surface" in tile:
                        tile["surface"].hide_render = True

                for additional_slab in additional_slabs:
                    additional_slab.hide_render = True

            added_objects = []

            if "mask" in mode:
                mask_cube = create_masked_cube(Vector((0,0,0)), height=1-z_offset-slab_height, z_offset=z_offset+slab_height)
                added_objects.append(mask_cube)

                for tile in tiles:
                    for lobj in get_self_and_children(tile):
                        if lobj and lobj.type == 'MESH':
                            lobj.data.materials.clear()
                            lobj.data.materials.append(holdout_material)
                            
                    if "surface" in tile:
                        tile["surface"].hide_render = True
                
                if not arg.no_slabs and has_next_slab_tile:
                    slab.data.materials.clear()
                    slab.data.materials.append(holdout_material)

                for additional_slab in additional_slabs:
                    additional_slab.data.materials.clear()
                    additional_slab.data.materials.append(holdout_material)

                for tile in tiles:
                    for obj in get_self_and_children(tile):
                        obj.hide_render = True

                for additional_slab in additional_slabs:
                    additional_slab.hide_render = True
    
                set_holdout_and_background("black")

            if "conditioning" in mode:
                # delete the mask_cube
                bpy.data.objects.remove(added_objects.pop())

                mask_slab = create_masked_cube(Vector((0,0,0)), height=1-z_offset, z_offset=z_offset)
                added_objects.append(mask_slab)

            if "ortho_scale" in view:
                cam.data.ortho_scale = view["ortho_scale"]
            
            cam.location = Vector((
                view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
                view['radius'] * np.sin(view['pitch'])
            ))
            
            cam.data.lens = 16 / np.tan(view['fov'] / 2)
            
            if arg.save_depth:
                spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
                spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
            
            bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}_{mode}.png')
            for name, output in outputs.items():
                output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')

            # Render the scene
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            for name, output in outputs.items():
                ext = EXT[output.format.file_format]
                path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
                os.rename(path, f'{output.file_slots[0].path}.{ext}')
                
            # Save camera parameters
            metadata = {
                "file_path": f'{i:03d}.png',
                "camera_angle_x": view['fov'],
                "transform_matrix": get_transform_matrix(cam)
            }
            if arg.save_depth:
                metadata['depth'] = {
                    'min': view['radius'] - 0.5 * np.sqrt(3),
                    'max': view['radius'] + 0.5 * np.sqrt(3)
                }
            to_export["frames"].append(metadata)

            # cleanup
            if "mask" in mode:
                for obj in added_objects:
                    bpy.data.objects.remove(obj)

                for tile in tiles:
                    for obj in get_self_and_children(tile):
                        obj.hide_render = False
                        
                    if "surface" in tile:
                        tile["surface"].hide_render = False

                for additional_slab in additional_slabs:
                    additional_slab.hide_render = False

                if not arg.no_slabs:
                    if has_next_slab_tile:
                        slab.hide_render = False
                    for additional_slab in additional_slabs:
                        slab.hide_render = True

                set_holdout_and_background("transparent")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, default=None, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--tiles', type=str, help='Comma-separated list of tile paths.')
    parser.add_argument('--next_tile_at', type=str, help='2D Vector indicating where to place a blank tile and center.')
    parser.add_argument('--skip_rendering_tiles', action='store_true', help='Whether the tiles will be rendered. Even if this is set to False, the tiles will be used to place the slab.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=1024, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--debase', action='store_true', help='If tiles have to be debaseted.')
    parser.add_argument('--cut_at', type=str, help='2D Vector indicating where to center a unit cube cut on the XY axes.')
    parser.add_argument('--no_slabs', action='store_true', help='Prevents slabs from being generated.')
    parser.add_argument('--rgb_only', action='store_true', help='Only render RGB.')
    parser.add_argument('--export_tile_info', action='store_true', help='Exports all tile information.')
    parser.add_argument('--no_render', action='store_true', help='Skips rendering.')
    parser.add_argument('--no_tile_modification', action='store_true', help='Skips tile modification.')
    parser.add_argument('--perspective_camera', action='store_true', help='Uses a perspective camera.')
    parser.add_argument('--no_lights', action='store_true', help='Disables lights.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    