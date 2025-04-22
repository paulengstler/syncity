import json
import os
import subprocess
from typing import Dict, List

os.environ['ATTN_BACKEND'] = 'xformers'

import dill
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from typing import Literal

import trellis.models as models
from tile_cutting import z_preserving_crop
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations.gaussian import Gaussian
from trellis.utils import render_utils

def generate_tile_info(blender_path: str, grid: List[Dict], output_folder: str, resolution: int = 1024):
    print("Generating tile info... ", end='', flush=True)
    # Construct command as a list of arguments
    cmd = [
        blender_path,
        '-b',
        '-P', 'blender_script.py',
        '--',
        '--output_folder', output_folder,
        '--resolution', str(resolution),
        '--debase',
        '--export_tile_info',
        '--no_render',
    ]

    if len(grid) > 0:
        tile_json = json.dumps(grid)
        cmd.extend(['--tiles', tile_json])

    # Run command with redirected output
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(cmd, stdout=devnull, stderr=devnull)

    print("done.")

def load_gaussian_from_tile(tile):
    # Loads a gaussian from a tile dictionary. Applies all transformations needed
    x, y, rotation, path = [tile[k] for k in ('x', 'y', 'rotation', 'path')]
    scale_factor, translation = [tile[k] for k in ('scale_factor', 'translation')]
    plane_cos, plane_nos = [tile[k] for k in ('plane_cos', 'plane_nos')]

    rgb_cut, z_preserving_cut, avg_z_val = [tile[k] for k in ('rgb_cut', 'z_preserving_cut', 'avg_z')]
    path = path.replace('tile.glb', 'mesh.ply')

    g = Gaussian(aabb=[-1, -1, -1, 1, 1, 1])
    g.load_ply(path, transform=None)
       
    min_x_rgb, max_x_rgb = rgb_cut['min_x'], rgb_cut['max_x']
    min_y_rgb, max_y_rgb = rgb_cut['min_y'], rgb_cut['max_y']

    min_x_z, max_x_z = z_preserving_cut['min_x'], z_preserving_cut['max_x']
    min_y_z, max_y_z = z_preserving_cut['min_y'], z_preserving_cut['max_y']

    g = g.crop(x_range=(min_x_z, max_x_z), y_range=(min_y_z, max_y_z))
    g = z_preserving_crop(g, x_range=(min_x_rgb, max_x_rgb), y_range=(min_y_rgb, max_y_rgb), z_range=(avg_z_val, 0.5))

    g.change_scale(scale_factor, center=(0, 0, 0))

    # coordinate conventions
    g.translate(np.array(translation) * np.array([-1, -1, 1]).tolist())

    # rotate
    g.rotate("Z", rotation, degrees=True)

    # print min and max for x and y
    return g
    
def get_next_tile(grid_info, x, y, dir):
    # Returns the tile in the grid_info list that is adjacent to the tile at (x, y) in the direction dir

    if dir == 'x':
        for tile in grid_info:
            if tile['x'] == x and tile['y'] == y + 1:
                return tile
    elif dir == 'y':
        for tile in grid_info:
            if tile['x'] == x + 1 and tile['y'] == y:
                return tile
    return None 

def get_conditioning_view(g1, g2, direction, view_type='frontal'):
    # Returns a view of g1 stitched to g2. The stitching is done in the direction specified by direction
    assert direction in ['x', 'y']
    if direction == 'x':
        g1.translate_y(-0.5)
        g2.translate_y(0.5)
    else:
        g1.translate_x(-0.5)
        g2.translate_x(0.5)
    g1.combine([g2])
    if view_type != 'zoom_out':
        g1.crop(
            x_range=(-0.5, 0.5),
            y_range=(-0.5, 0.5),
            z_range=None
        )
    if view_type == 'frontal' or view_type == 'zoom_out':
        image = render_utils.render_single_frame(
            g1, 
            yaw=90 if direction == 'x' else 180, 
            pitch=0.5,
            resolution=1024,
            r=2 if view_type!= 'zoom_out' else 3
            )['color'][0]
    elif view_type == 'aerial':
        image = render_utils.render_single_frame(
            g1, 
            yaw=90 if direction == 'x' else 180, 
            pitch=np.pi/2,
            resolution=1024,
            r=2 if view_type!= 'zoom_out' else 3
            )['color'][0]
    return image 

def get_conditioning_slats(s1, s2, direction, border=2, cut=0):
    from trellis.modules import sparse as sp
    # Returns a view of g1 stitched to g2. The stitching is done in the direction specified by direction
    assert direction in ['x', 'y']
    if direction == 'y':
        s1_keep = torch.logical_and(s1.coords[:, 1] >=32, s1.coords[:, 1] < 64 - cut)
        s1_coords = s1.coords[s1_keep] - torch.tensor([0, 32 - cut, 0, 0]).cuda()
        s1_feats = s1.feats[s1_keep]

        s2_keep = s2.coords[:, 1] < 32 - cut
        s2_coords = s2.coords[s2_keep] + torch.tensor([0, 32, 0, 0]).cuda()
        s2_feats = s2.feats[s2_keep]

    else:
        s1_keep = torch.logical_and(s1.coords[:, 2] >=32, s1.coords[:, 2] < 64 - cut)
        s1_coords = s1.coords[s1_keep] - torch.tensor([0, 0, 32 - cut, 0]).cuda()
        s1_feats = s1.feats[s1_keep]

        s2_keep = s2.coords[:, 2] < 32 - cut
        s2_coords = s2.coords[s2_keep] + torch.tensor([0, 0, 32, 0]).cuda()
        s2_feats = s2.feats[s2_keep]
    coords = torch.cat([s1_coords, s2_coords], dim=0).type(torch.int32)
    feats = torch.cat([s1_feats, s2_feats], dim=0)

    slat_combined = sp.SparseTensor(
        feats=feats.cuda(),
        coords=coords.cuda()
    )

    feats_cond = torch.zeros_like(feats).cuda()

    if direction == 'y':
        coords_to_change = torch.logical_and(coords[:, 1] >= 32 - border, coords[:, 1] < 32 + border)
        coords_to_change = torch.logical_or(coords_to_change, coords[:, 1] >= 58)
        coords_to_change = torch.logical_or(coords_to_change, coords[:, 1] < 5)
    else:
        coords_to_change = torch.logical_and(coords[:, 2] >= 32 - border, coords[:, 2] < 32 + border)
        coords_to_change = torch.logical_or(coords_to_change, coords[:, 2] >= 58)
        coords_to_change = torch.logical_or(coords_to_change, coords[:, 2] < 5)

    feats_cond[coords_to_change] = 1

    slat_mask = sp.SparseTensor(
        feats=feats_cond,
        coords=coords.cuda()
    )

    return slat_combined, slat_mask

def to_idx(value: float, num_bins: int = 64, min_val: float = -0.5, max_val: float = 0.5) -> int:
    bin_width = (max_val - min_val) / num_bins
    idx = int((value - min_val) / bin_width)
    return np.clip(idx, 0, num_bins - 1)

def largest_black_region_mask_pil(pil_image, min_area=500):
    import cv2
    # Convert PIL image to grayscale NumPy array
    image = np.array(pil_image.convert("L"))  # Convert to grayscale
    
    # Create a binary mask where black pixels (close to 0) are white (255) and others are black (0)
    _, binary_mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological opening to remove small black noise
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)

    if num_labels <= 1:
        return Image.fromarray(np.zeros_like(image, dtype=np.uint8))  # No valid regions found
    
    # Ignore background (label 0) and find the largest component by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    # Create a mask for the largest black region
    largest_mask = (labels == largest_label).astype(np.uint8) * 255

    # Convert NumPy array back to a PIL image
    return Image.fromarray(largest_mask)

def process_images_for_pipeline(images, pipeline, fake_alpha=False, uncond=False):
    if not isinstance(images, list):
        images = [images]
    if not isinstance(images[0], Image.Image):
        images = [Image.fromarray(image) for image in images]
    if fake_alpha:
        # alpha is the black pixels

        image_np = np.array(images[0])
        mask = largest_black_region_mask_pil(images[0])
        mask_np = 255 - np.array(mask)        
        # Create RGBA image by adding the alpha channel
        rgba_np = np.dstack([image_np, mask_np])

        # Convert back to PIL image
        rgba_image = Image.fromarray(rgba_np, mode="RGBA")

        images = [rgba_image]

    processed_images = [pipeline.preprocess_image(image, center_input=False) for image in images]
    cond = pipeline.get_cond(processed_images)
    cond = {k: v.cuda() for k, v in cond.items()}
    if uncond:
        cond['cond'] = cond['neg_cond']
    return cond

def structure_preserving_interpolation(voxels, scale_factor, method='nearest_then_binary'):
    """
    Interpolate 3D voxel data while preserving structural integrity.
    
    Args:
        voxels (torch.Tensor): Input voxel grid with shape [D, H, W] or [D, H, W, C]
        scale_factor (float or tuple): Scale factor for each dimension (z, y, x)
        method (str): Interpolation method:
            - 'nearest_then_binary': Uses nearest neighbor interpolation then applies threshold
            - 'binary_dilation_erosion': Performs binary dilation followed by erosion
            - 'distance_field': Uses distance field transform for interpolation
            - 'trilinear_high_threshold': Uses trilinear interpolation with higher threshold
            - 'occupancy_based': Preserves original occupancy rate
            
    Returns:
        torch.Tensor: Interpolated voxel grid
    """
    has_channels = len(voxels.shape) == 4
    
    if has_channels:
        D, H, W, C = voxels.shape
        voxels_input = voxels.permute(3, 0, 1, 2)  # [C, D, H, W]
    else:
        D, H, W = voxels.shape
        voxels_input = voxels.unsqueeze(0)  # [1, D, H, W]
    
    if isinstance(scale_factor, (list, tuple)):
        scale_h, scale_w, scale_d = scale_factor
    
    new_D, new_H, new_W = min(int(D * scale_h), 64), min(int(H * scale_w), 64), min(int(W * scale_d), 64)
    
    # Method 1: Nearest-neighbor interpolation followed by binary threshold
    if method == 'nearest_then_binary': # NOTE: Seems to work best on my examples so far... if any issues, revisit
        # Use nearest neighbor to avoid blurring boundaries
        resized = F.interpolate(
            voxels_input.unsqueeze(0),  # [1, C, D, H, W] or [1, 1, D, H, W]
            size=(new_D, new_H, new_W),
            mode='nearest'
        ).squeeze(0)
        
        # Apply threshold to ensure binary voxels (if input was binary)
        if voxels.dtype == torch.bool or ((voxels == 0) | (voxels == 1)).all():
            resized = resized > 0.5
    else:
        raise ValueError(f"Invalid method: {method}")
    
    # Format output to match input tensor shape
    if has_channels:
        resized = resized.permute(1, 2, 3, 0)  # [D, H, W, C]
    else:
        resized = resized.squeeze(0)  # [D, H, W]
        
    return resized

def get_mask(image, pts = (0.4, 0.48, 0.75, 0.49)):
    from PIL import ImageDraw
    x, y = image.size
    x1, x2, y1, y2 = pts
    # Create a blank grayscale (L-mode) image
    mask = Image.new("L", (x, y), 0)


    # Define the parallelogram points
    pts = [
        (x1 * x, y1 * y), 
        (x2 * x, y1 * y), 
        (x2 * x, y2 * y),
        (x1 * x, y2 * y)]

    # Draw the filled parallelogram
    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, fill=255)

    return mask

def sample_slat(voxels_grid, views, pipeline, render_video=False):
    if len(voxels_grid.shape) == 3:
        voxels_grid = voxels_grid.unsqueeze(0).unsqueeze(0)
    coords = torch.argwhere(voxels_grid > 0)[:, [0, 2, 3, 4]].int()
    cond = process_images_for_pipeline(views, pipeline)
    cond['neg_cond'] = cond['neg_cond'][:1]
    slat, slat_raw = pipeline.sample_slat_multi_image(coords.cuda(), cond)
    gs = pipeline.decode_slat(slat, ['gaussian'])
    gs = gs['gaussian'][0]
    output = {
        'slat' : slat,
        'slat_raw' : slat_raw,
        'gs' : gs
    }
    if render_video:
        video = render_utils.render_video(gs, num_frames=300, resolution=512, r=2)['color']
        output['video'] = video
    return output

def sample_slat_cond(views, pipeline, cond_slat, cond_slat_mask, render_video=False):
    cond = process_images_for_pipeline(views, pipeline)
    cond['neg_cond'] = cond['neg_cond'][:1]
    slat, slat_raw = pipeline.sample_slat_multi_image(cond_slat.coords, cond, {}, cond_samples=cond_slat, cond_samples_mask=cond_slat_mask)
    gs = pipeline.decode_slat(slat, ['gaussian'])
    gs = gs['gaussian'][0]
    output = {
        'slat' : slat,
        'slat_raw' : slat_raw,
        'gs' : gs
    }
    if render_video:
        video = render_utils.render_video(gs, num_frames=300, resolution=512, r=2)['color']
        output['video'] = video
    return output
    
def rotate_voxel_grid_z(voxels_grid, rotation_degrees):
    # Validate rotation angle
    if rotation_degrees not in [0, 90, 180, 270]:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")
    
    # If no rotation, return the original
    if rotation_degrees == 0:
        return voxels_grid
    
    # Implement rotations using vectorized operations
    if rotation_degrees == 90:
        # 90 degrees clockwise around z: (x, y) -> (y, -x)
        # Transpose first two dims, then flip the second dim
        return voxels_grid.transpose(0, 1).flip(0)
    
    elif rotation_degrees == 180:
        # 180 degrees around z: (x, y) -> (-x, -y)
        # Flip both dimensions
        return voxels_grid.flip(0).flip(1)
    
    elif rotation_degrees == 270:
        # 270 degrees clockwise around z: (x, y) -> (-y, x)
        # Transpose first two dims, then flip the first dim
        return voxels_grid.transpose(0, 1).flip(1)
    
@torch.no_grad()
def get_rescaled_cropped_slat(tile, views, z, pipeline, ss_decoder):
    from trellis.modules import sparse as sp

    tile_trellis = torch.load(tile['path'].replace('tile.glb', 'tile.pt'))

    ss_decoded = ss_decoder(tile_trellis['ss_latents'])

    x, y, rotation, path = [tile[k] for k in ('x', 'y', 'rotation', 'path')]
    print(f'Processing tile at ({x}, {y})...')  

    rgb_cut, z_preserving_cut, avg_z_val = [tile[k] for k in ('rgb_cut', 'z_preserving_cut', 'avg_z')]

    min_x_z_, max_x_z_ = z_preserving_cut['min_x'], z_preserving_cut['max_x']
    min_x_z = to_idx(min_x_z_)
    max_x_z = to_idx(max_x_z_)
    min_y_z_, max_y_z_ = z_preserving_cut['min_y'], z_preserving_cut['max_y']
    min_y_z = to_idx(min_y_z_)
    max_y_z = to_idx(max_y_z_)    

    ss_decoded = ss_decoded[0, 0, min_x_z : max_x_z+1, min_y_z : max_y_z+1, :]

    min_z, max_z = torch.argwhere(ss_decoded > 0)[:, 2].min(), torch.argwhere(ss_decoded > 0)[:, 2].max()
    ss_decoded = ss_decoded[:, :, min_z:max_z]

    # check max and min in x and y
    coords = torch.argwhere(ss_decoded > 0).int()
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    min_z, max_z = coords[:, 2].min(), coords[:, 2].max()

    scale_x = 63 / (max_x - min_x) 
    scale_y = 63 / (max_y - min_y)
    scale_z = min(64-17,(to_idx(z[1]) - to_idx(z[0])))/ (max_z - min_z) # somewhat ugly hack
    # scale_z = (scale_x + scale_y) / 2
    ss_decoded = structure_preserving_interpolation(
        ss_decoded, 
        (
            scale_x,
            scale_y,
            scale_z
        )) > 0
    
    border_tiles = ss_decoded.sum(0).sum(0) - ss_decoded[5:63-5,5:63-5, :].sum(0).sum(0)
    border_tiles[:4] = 0 # skip the first few..
    max_value = border_tiles.max()
    is_base = border_tiles > max_value * 0.75
    # find max index where true
    max_idx = torch.argmax(is_base * torch.arange(len(is_base)).cuda())
    # fully extend the base
    ss_decoded[:, :, max_idx] = 1 

    is_full = ss_decoded[:,:,max_idx]
    ss_decoded_full = ss_decoded.clone()
    ss_decoded_full_diff = torch.zeros_like(ss_decoded)
    for i in range(max_idx):
        idx = max_idx - i - 1
        is_full = torch.logical_or(is_full, ss_decoded[:,:,idx]) * 1
        ss_decoded_full[:,:,idx] = is_full
        ss_decoded_full_diff[:,:,idx] = is_full - ss_decoded[:,:,idx] * 1
    voxels_grid = torch.zeros(64, 64, 64)
    # put max_idx at 17 -- somewhat arbitrary
    z_move = 17 - max_idx
    z_move = max(0, z_move)

    # voxels_grid[:, :, to_idx(z[0]) :to_idx(z[0]) + ss_decoded.shape[2]] = ss_decoded
    voxels_grid[:, :, z_move: z_move + ss_decoded.shape[2]] = ss_decoded
    voxels_grid = rotate_voxel_grid_z(voxels_grid, rotation)
    
    # First denoise original. The add more voxels and denoise them on top.
    output_sparse = sample_slat(voxels_grid, views, pipeline, render_video=True)
    
    slat_cond_raw_sparse = output_sparse['slat_raw']

    feats_sparse = slat_cond_raw_sparse.feats
    coords_sparse = slat_cond_raw_sparse.coords

    coords_to_full = torch.argwhere(ss_decoded_full_diff.unsqueeze(0).unsqueeze(0) > 0)[:, [0, 2, 3, 4]].int().cuda() + torch.tensor([0, 0, 0, to_idx(z[0])]).int().cuda()
    feats_to_full = torch.zeros(coords_to_full.shape[0], feats_sparse.shape[1]).cuda()

    feats_full = torch.cat([feats_sparse, feats_to_full], dim=0)
    coords_full = torch.cat([coords_sparse, coords_to_full], dim=0) 

    slat_cond_full = sp.SparseTensor(
        feats=feats_full,
        coords=coords_full
    )

    feats_mask = torch.cat([torch.zeros_like(feats_sparse), torch.ones_like(feats_to_full)], dim=0).cuda()

    slat_cond_full_mask = sp.SparseTensor(
        feats=feats_mask,
        coords=coords_full
    )

    output_full = sample_slat_cond(views, pipeline, slat_cond_full, slat_cond_full_mask)

    return {
        'voxels' : voxels_grid,
        'slat' : output_full['slat'],
        'slat_raw' : output_full['slat_raw'],
        'gs' : output_full['gs']
    }

@torch.no_grad()
def merge_gaussians(
        prefix: str,
        compute_rescaled: bool = False, 
        stitch_images: bool = False, 
        stitch_slats: bool = False, 
        use_cached: bool = False,
        inpainter_type: Literal["flux_local", "flux_replicate", "sdxl_replicate"] = "flux_local",
        gradio_url='http://127.0.0.1:7860',
        blender_path: str = 'blender-3.6.19-linux-x64/blender',
        seed: int = 429
    ):

    grid_path = prefix
    print(f'Options are compute_rescaled={compute_rescaled}, stitch_images={stitch_images}, stitch_slats={stitch_slats}')
    grid_info_path = os.path.join(grid_path, 'grid.json')

    grid_info = json.load(open(grid_info_path))
    xs = [g['x'] for g in grid_info]
    ys = [g['y'] for g in grid_info]

    generate_tile_info(blender_path, grid_info, grid_path)
    
    # augment the grid_info with the new tile info
    tile_info_path = os.path.join(grid_path, 'tile_info.json')
    tile_info = json.load(open(tile_info_path))

    # sort both
    grid_info = sorted(grid_info, key=lambda x: (x['x'], x['y']))
    tile_info = sorted(tile_info, key=lambda x: (x['x'], x['y']))

    for i, g in enumerate(grid_info):
        assert g['x'] == tile_info[i]['x']
        assert g['y'] == tile_info[i]['y']

        g.update(tile_info[i])

    # find min and max x and y
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    extent_x, extent_y = max_x - min_x + 1, max_y - min_y + 1
    max_extent = max(extent_x, extent_y)

    # find the center of the grid
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    scene_vanilla = None
    scene = None

    ss_decoder = models.from_pretrained("JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16").eval().cuda()
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()

    rescaled_tiles = {}
    from copy import deepcopy
    if compute_rescaled:
        for tile in tqdm(grid_info, desc="Computing rescaled tiles..."):
            x, y, rotation, path = [tile[k] for k in ('x', 'y', 'rotation', 'path')]
            g = load_gaussian_from_tile(tile)

            views = render_utils.render_mv_camera_sequence(g)['color']

            z = (g.get_xyz[:, 2].min(), g.get_xyz[:, 2].max())
            rescaled_tiles[f'{x}_{y}'] = get_rescaled_cropped_slat(tile, views, z, pipeline, ss_decoder)

            g_ = deepcopy(rescaled_tiles[f'{x}_{y}']['gs'])
            video = render_utils.render_video(g_, num_frames=50, resolution=512)['color']
            imageio.mimsave(f'{grid_path}/{x},{y}/recon.mp4', video, fps=5)

            g.translate((x - center_x, y - center_y, 0))

            g_.translate((x - center_x, y - center_y, 0))

            if scene_vanilla is None:
                scene_vanilla = g
            else:
                scene_vanilla.combine([g])
                print(scene_vanilla.get_xyz.shape)

            del g_

        dill.dump(rescaled_tiles, open(os.path.join(grid_path, 'rescaled_tiles.pkl'), 'wb'))
    
        for k in list(pipeline.models.keys()):
            del pipeline.models[k]

        del pipeline.sparse_structure_sampler
        del pipeline.slat_sampler

        del pipeline

        del ss_decoder

        torch.cuda.empty_cache()

        video = render_utils.render_video(scene_vanilla, num_frames=300, resolution=2048, r=max_extent*2.5)['color']
        imageio.mimsave(os.path.join(grid_path, "gaussians.mp4"), video, fps=30)

        del scene_vanilla

        torch.cuda.empty_cache()
        
    else:
        rescaled_tiles = dill.load(open(os.path.join(grid_path, 'rescaled_tiles.pkl'), 'rb'))

    if stitch_images:
        from inpainting import Inpainter
        import time
        inpainter = Inpainter(inpainter_type, gradio_url)

        VIEW_TYPE = 'zoom_out'
        prompts = json.load(open(os.path.join(grid_path, 'instructions.json')))
        for tile in tqdm(grid_info, desc="Stitching images..."):
            x, y = [tile[k] for k in ('x', 'y')]
            prompt = [p['prompt'] for p in prompts if p['x'] == x and p['y'] == y][0]
            for direction in ['x', 'y']:
                if os.path.exists(f'{grid_path}/{x},{y}/stitch_to_{direction}_inpainted.png') and use_cached:
                    print(f'Skipping {x},{y} to {direction}...')
                    continue
                next_tile = get_next_tile(grid_info, x, y, direction)
                if next_tile is None:
                    continue
                g1 = deepcopy(rescaled_tiles[f'{x}_{y}']['gs'])
                g2 = deepcopy(rescaled_tiles[f'{next_tile["x"]}_{next_tile["y"]}']['gs'])
                view_np = get_conditioning_view(g1, g2, direction, VIEW_TYPE) # TODO: maybe use to condition denoising
                view = Image.fromarray(view_np)
                
                if VIEW_TYPE == 'frontal':
                    # This provides a close up of the scene. Does not work as well as zoom_out
                    view = view.crop((view.width * 0.13, view.height * 0.13, view.width * 0.87, view.height * 0.87))
                    mask = get_mask(view, (0.43 , 0.57, 0.45, 1.0))
                    prompt_combined = 'close up of a scene. infinite horizon, vast view to the back.'

                elif VIEW_TYPE == 'zoom_out':
                    mask_black = (np.array(get_mask(view, (0.0 , 1.0, 0.72, 0.95)))[:, :, None] > 0).astype(np.uint8)
                    # make these pixels black in the view
                    view_np = np.array(view)
                    view_np = (1 - mask_black) * view_np + mask_black * np.zeros_like(view_np)
                    view = Image.fromarray(view_np)
                    mask = get_mask(view, (0.43 , 0.57, 0.48, 0.75))
                    prompt_combined = "beautiful scene, well blended features and seamless continuity, realistic textures, soft diffused shading and well lit, subtle gradients, part of a city-building 3D game, meticulous detailing"

                view.save(f'{grid_path}/{x},{y}/stitch_to_{direction}.png')
                # try then wait 10 s then try again -- sometimes the gradio server times out if busy
                while True:
                    try:
                        inpainted = inpainter(view, mask.convert('L'), seed + x * 10 + y * 100, prompt_combined)
                        break
                    except:
                        time.sleep(10)
                if VIEW_TYPE == 'zoom_out':
                    inpainted = inpainted.crop((view.width * 0.25, view.height * 0.22, view.width * 0.75, view.height * 0.72))
                inpainted.save(f'{grid_path}/{x},{y}/stitch_to_{direction}_inpainted.png')
    
    scene = None
    BORDER = 4
    CUT = 0
    FRAC_TILE = 1.5
    FRAC_INPAINT = 2.0
    GRID=64
    UNCOND=False
    
    if stitch_slats:
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
        for tile in tqdm(grid_info, desc="Stitching slats"):
            x, y, = [tile[k] for k in ('x', 'y')]

            slats = rescaled_tiles[f'{x}_{y}']['slat_raw']

            g = deepcopy(rescaled_tiles[f'{tile["x"]}_{tile["y"]}']['gs'])
            g_xmin = g.get_xyz[:, 0].min()
            g_xmax = g.get_xyz[:, 0].max()
            g_ymin = g.get_xyz[:, 1].min()
            g_ymax = g.get_xyz[:, 1].max()
            x0 = g.get_xyz[:, 0].min() if x == 0 else g.get_xyz[:, 0].min() + BORDER * FRAC_TILE / GRID + (CUT / GRID)
            x1 = g.get_xyz[:, 0].max() if x == max_x else g.get_xyz[:, 0].max() - BORDER * FRAC_TILE / GRID - (CUT / GRID)
            y0 = g.get_xyz[:, 1].min() if y == 0 else g.get_xyz[:, 1].min() + BORDER * FRAC_TILE / GRID + (CUT /GRID)
            y1 = g.get_xyz[:, 1].max() if y == max_y else g.get_xyz[:, 1].max() - BORDER * FRAC_TILE / GRID - (CUT /GRID)
            g.crop(x_range=(x0, x1), y_range=(y0, y1), z_range=None)
            g.translate((x - center_x, y - center_y, 0))
            g.translate((- 2 * x * CUT  / GRID, - 2 * y * CUT / GRID, 0))
            
            if scene is None:
                scene = deepcopy(g)
            else:
                scene.combine([g])

            for direction in ['x', 'y']:
                if os.path.exists(f'{grid_path}/{x},{y}/stitch_to_{direction}_slat.ply') and use_cached:
                    output_gs = Gaussian(aabb=[-1, -1, -1, 1, 1, 1])
                    output_gs.load_ply(f'{grid_path}/{x},{y}/stitch_to_{direction}_slat.ply', transform=None)
                else:
                    next_tile = get_next_tile(grid_info, x, y, direction)
                    if next_tile is None:
                        continue
                    slats_next = rescaled_tiles[f'{next_tile["x"]}_{next_tile["y"]}']['slat_raw']
                    slat_combined, slat_mask = get_conditioning_slats(slats, slats_next, direction, border=BORDER, cut=CUT)
                    cond_image = Image.open(f'{grid_path}/{x},{y}/stitch_to_{direction}_inpainted.png')
                    with torch.no_grad():
                        cond = process_images_for_pipeline(cond_image, pipeline, fake_alpha=True, uncond=UNCOND)
                        slat, slat_raw = pipeline.sample_slat(cond, slat_combined.coords, {}, slat_combined, slat_mask)
                        output_gs = pipeline.decode_slat(slat, ['gaussian'])['gaussian'][0]
                    output_gs.save_ply(f'{grid_path}/{x},{y}/stitch_to_{direction}_slat.ply', transform=None)

                if direction == 'x':
                    x0 = g_xmin if x == 0 else g_xmin #+ (BORDER / GRID) / 2
                    x1 = g_xmax if x == max_x else g_xmax #- (BORDER / GRID) / 2 
                    y0 = - BORDER * FRAC_INPAINT / GRID 
                    y1 =   BORDER * FRAC_INPAINT / GRID 
                else:
                    x0 = - BORDER * FRAC_INPAINT / GRID 
                    x1 =   BORDER * FRAC_INPAINT / GRID 
                    y0 = g_ymin if y == 0 else g_ymin #+ (BORDER / GRID) / 2
                    y1 = g_ymax if y == max_y else g_ymax #- (BORDER / GRID) / 2

                output_gs.crop(x_range=(x0, x1), y_range=(y0, y1), z_range=None)
                translate_y = 0.5 * (1 if direction == 'x' else 0) #- ( 1 / GRID if direction == 'x' else 0)
                translate_x = 0.5 * (1 if direction == 'y' else 0) #- ( 1 / GRID if direction == 'y' else 0)
                output_gs.translate((translate_x, translate_y, 0))
                output_gs.translate((x - center_x, y - center_y, 0))
                output_gs.translate(
                    (- (2 * x) * CUT / 64 if direction == 'x' else - (2 * x + 1) * CUT / 64,
                    - (2 * y ) * CUT / 64 if direction == 'y' else - (2 * y + 1) * CUT / 64,
                    0))
                
                scene.combine([output_gs])
   
                torch.cuda.empty_cache()

        # free cuda memory from pipeline
        for k in list(pipeline.models.keys()):
            del pipeline.models[k]

        del pipeline.sparse_structure_sampler
        del pipeline.slat_sampler

        del pipeline
        torch.cuda.empty_cache()

        scene.save_ply(f'{grid_path}/gaussians_scene.ply', transform=None)

        video = render_utils.render_video(scene, num_frames=300, resolution=2048, r=int(max_extent*2.5), pitch_mean=0.3, pitch_offset=0.15, bg_color=(1,1,1))['color']
        imageio.mimsave(f'{grid_path}/gaussians_blended.mp4', video, fps=30)

if __name__ == '__main__':
    import fire
    fire.Fire(merge_gaussians)
