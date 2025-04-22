import glob
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from enum import Enum
from functools import partial
from queue import Queue
from typing import Dict, List, Literal, Optional, Tuple

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"  # 'auto' is faster but benchmarks on start

import cv2
import imageio
import numpy as np
import PIL.ImageOps
import rembg
from easydict import EasyDict as edict
from lpips import LPIPS, im2tensor
from PIL import Image

from inpainting import Inpainter

# a brief explanation of the orthographic scale:
# the ortho scale determines the size of the world in the image
# a larger ortho scale means a smaller world, meaning we see more context
# as we build the grid, we want to see more context, thus we
# increase the ortho scale for the first rows (up to a certain limit)

# this is "desired" ortho scale of the conditioning images for TRELLIS
# that means, every tile will have the exact same size in the conditioning image
BASE_ORTHO_SCALE = 1.75

# this is the ortho scale which we'll use for the first row (except the first one, which will get the BASE_ORTHO_SCALE)
INITIAL_ORTHO_SCALE = 2.0
ORTHO_SCALE_Y_STEP = 0.5
ORTHO_SCALE_MAX_NUM_Y_STEPS = 2

# this means we'll max out at 2.0 + 2 * 0.5 = 3.0
MAX_ORTHO_SCALE = INITIAL_ORTHO_SCALE + ORTHO_SCALE_Y_STEP * ORTHO_SCALE_MAX_NUM_Y_STEPS


class States(Enum):
    BLOCKED = "🔒"
    READY = "📦"
    ASSIGNED = "📋"
    RENDERING = "🎥"
    STALLED = "⏳"
    INPAINTING = "🎨"
    REBASING = "🪏"
    GENERATING = "🏗️"
    CUTTING = "📐"
    ORIENTING = "🧭"
    DONE = "✅"
    CRASH = "💥"


def process_instructions(instructions_path: str) -> List[str]:
    with open(instructions_path, "r") as f:
        instructions = json.load(f)

    # sort the instructions by the x and y values
    instructions["tiles"] = sorted(
        instructions["tiles"], key=lambda x: (x["y"], x["x"])
    )

    assert (
        instructions["tiles"][0]["x"] == 0 and instructions["tiles"][0]["y"] == 0
    ), "The first tile must be at position (0, 0)"

    for tile_instruction in instructions["tiles"]:
        tile_instruction["prompt"] = instructions["prompt"].format(
            **{"tile_prompt": tile_instruction["prompt"]}
        )

    return instructions


def label_obstructing_tiles(
    grid: List[Dict],
    tile_pos: Tuple[int],
    key: str,
    create_copy: bool = True,
    default_slice_height=0.0,
):
    if create_copy:
        grid = json.loads(json.dumps(grid))

    for tile in grid:
        if tile["x"] == tile_pos[0] and tile["y"] == tile_pos[1] - 1:
            if "max_corner" in tile:
                tile[key] = tile["max_corner"] + tile["translation"][-1] + 5e-3
            else:
                tile[key] = default_slice_height

    return grid


def make_non_overlapping_mask(
    mask,
    full_mask: bool = False,
    overlap_x: int = 8,
    overlap_y: int = 8,
    extend_at: Optional[Literal["x", "y", "xy"]] = "xy",
    add_corner: bool = True,
    erosion_radius: int = 5,
):
    mask_np = np.array(mask)

    # erode the mask by a tiny amount
    if erosion_radius > 0:
        cv2.erode(mask_np, np.ones((erosion_radius, erosion_radius), np.uint8), mask_np)

    # Convert the mask to a binary version (0 and 1) for easier processing.
    bin_mask = (mask_np > 0).astype(np.uint8)

    # Sum along the vertical direction (axis=0) to get a 1D array per column.
    col_sums = bin_mask.sum(axis=0)

    # Find the first column that contains any white pixels.
    nonzero_cols = np.where(col_sums > 0)[0]
    if len(nonzero_cols) == 0:
        raise ValueError("The mask does not contain any white pixels.")
    x1 = nonzero_cols[0]

    # The other edge is the column with the maximum white pixels.
    x2 = int(np.argmax(col_sums))

    # For column x1, get the indices (rows) where pixels are white.
    rows_x1 = np.where(bin_mask[:, x1] > 0)[0]
    y1_top, y1_bot = int(rows_x1[0]), int(rows_x1[-1])

    # For column x2, get the indices (rows) where pixels are white.
    rows_x2 = np.where(bin_mask[:, x2] > 0)[0]
    y2_top, y2_bot = int(rows_x2[0]), int(rows_x2[-1])

    # Create a copy of the mask to modify.
    mask_out = mask_np.copy()
    d = 5  # fix white boundaries
    h = abs(y1_top - y1_bot) - (
        0 if not extend_at or "x" not in extend_at else overlap_x
    )

    if not full_mask:
        # Define the four corners of the quadrilateral (in (x, y) order):
        # Top-left, top-right, bottom-right, bottom-left.
        pts = np.array(
            [
                [x1, y1_top - d],
                [x2, y2_top - d],
                [x2, y2_top + h - d],
                [x1, y1_top + h - d],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask_out, [pts], 0)

    bot_slope = (y2_bot - y1_bot) / (x2 - x1)
    top_slope = (y2_top - y1_top) / (x2 - x1)

    if add_corner and extend_at and "x" in extend_at:
        # we'll add a small corner to the left, so Flux can erase
        # discontinuities in the x direction

        b_bot = (y1_bot) - (bot_slope * x1)
        b_top = (y1_top + h - d) - (top_slope * x1)

        x_intersect = (b_top - b_bot) / (bot_slope - top_slope)
        y_intersect = bot_slope * x_intersect + b_bot

        x_int, y_int = int(x_intersect), int(y_intersect)

        pts = np.array(
            [[x1, y1_bot], [x_int, y_int], [x1, y1_top + h - d]], dtype=np.int32
        )

        cv2.fillPoly(mask_out, [pts], 1)

    if extend_at and "y" in extend_at:
        pts = np.array(
            [
                [x1, y1_bot],
                [x2, y2_bot],
                [x2 - overlap_y, y2_bot + overlap_y * -top_slope],
                [x1 - overlap_y, y1_bot + overlap_y * -top_slope],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask_out, [pts], 1)

    return mask_out


def generate_tile_info(
    blender_path, grid: List[Dict], output_folder: str, resolution: int = 1024
):
    grid = json.loads(json.dumps(grid))

    # Construct command as a list of arguments
    cmd = [
        blender_path,
        "-b",
        "-P",
        "blender_script.py",
        "--",
        "--output_folder",
        output_folder,
        "--resolution",
        str(resolution),
        "--debase",
        "--export_tile_info",
        "--no_render",
    ]

    if len(grid) > 0:
        tile_json = json.dumps(grid)
        cmd.extend(["--tiles", tile_json])

    # Run command with redirected output
    with open(os.devnull, "wb") as devnull:
        subprocess.check_call(cmd, stdout=devnull, stderr=devnull)


def render_next_tile(
    blender_path,
    grid: List[Dict],
    output_folder: str,
    resolution: int = 1024,
    pos: Tuple[int, int] = (0, 0),
    ortho_scale: float = 1.75,
):
    grid = json.loads(json.dumps(grid))

    # figure out which tiles are at most 2 tiles away (Manhattan distance) from the current tile
    # and only render tiles that aren't above the current tile (which would mess with the mask)
    grid = [
        tile
        for tile in grid
        if tile["x"] <= pos[0]
        and tile["y"] <= pos[1]
        and not (tile["x"] == pos[0] and tile["y"] == pos[1])
    ]

    # to provide additional context for tiles x=0, we put the y-1 tile (if it exists)
    # at position (-1, y) as this will not be cropped out and can provide additional context
    if pos[0] == 0 and pos[1] > 0:
        # find the tile at y-1
        y_minus_1 = [
            tile for tile in grid if tile["x"] == 0 and tile["y"] == pos[1] - 1
        ]
        if len(y_minus_1) > 0:
            tile_dict = y_minus_1[0]
            grid.append({**tile_dict, "x": -1, "y": pos[1], "has_slab": False})

    label_obstructing_tiles(grid, pos, "slice_z", create_copy=False)

    # Construct command as a list of arguments
    cmd = [
        blender_path,
        "-b",
        "-P",
        "blender_script.py",
        "--",
        "--output_folder",
        output_folder,
        "--resolution",
        str(resolution),
        "--debase",
        f"--next_tile_at={pos[0]},{pos[1]}",
    ]

    if len(grid) > 0:
        cmd.extend(["--tiles", json.dumps(grid)])

    views = [
        {
            "yaw": np.radians(-45),
            "pitch": np.arctan(1 / np.sqrt(2)),
            "radius": 2,
            "fov": np.radians(47.1),
            "ortho_scale": ortho_scale,
        }
    ]
    cmd.extend(["--views", json.dumps(views)])

    # Run command with redirected output
    with open(os.devnull, "wb") as devnull:
        subprocess.check_call(cmd, stdout=devnull, stderr=devnull)


def find_orientation_of_tile(
    blender_path,
    tile_dict: Dict,
    conditioning_image: str,
    output_folder: str,
    resolution: int = 256,
    rotations: Tuple[int] = (0, 90, 180, 270),
):
    # place this singular tile at the origin
    tile_dict = json.loads(json.dumps(tile_dict))
    tile_dict["x"], tile_dict["y"] = 0, 0

    for rotation in rotations:
        tile_dict["rotation"] = rotation
        tile_json = json.dumps(
            [tile_dict]
        )  # No need to escape quotes when using list arguments

        cmd = [
            blender_path,
            "-b",
            "-P",
            "blender_script.py",
            "--",
            "--output_folder",
            os.path.join(output_folder, f"rot_{rotation}"),
            "--resolution",
            str(resolution),
            "--tiles",
            tile_json,
            "--rgb_only",
        ]

        # Run command with redirected output
        with open(os.devnull, "wb") as devnull:
            subprocess.check_call(cmd, stdout=devnull, stderr=devnull)

    # find the orientation of the tile
    available_rotations = glob.glob(f"{output_folder}/rot_*/*.png")
    rotations_dict = {
        int(fn.split("rot_")[-1].split("/")[0]): Image.open(fn).convert("RGBA")
        for fn in available_rotations
    }

    conditioning = (
        Image.open(conditioning_image).convert("RGBA").resize((resolution, resolution))
    )
    lpips_inp_cond = im2tensor(np.array(conditioning.convert("RGB"))[:, :, ::-1]).to(
        "cuda"
    )

    lpips_fn = LPIPS(net="vgg").cuda()
    lpips_loss = {
        rotation: lpips_fn(
            lpips_inp_cond,
            im2tensor(np.array(rotations_dict[rotation].convert("RGB"))[:, :, ::-1]).to(
                "cuda"
            ),
        ).item()
        for rotation in rotations_dict
    }

    # clean up
    for rotation in rotations_dict:
        shutil.rmtree(f"{output_folder}/rot_{rotation}")

    return min(lpips_loss, key=lpips_loss.get)


def process_mask(mask_image: Image) -> Image:
    return Image.fromarray(
        (np.floor(np.array(mask_image.convert("L")) / 255)).clip(0, 1).astype(np.uint8)
        * 255
    )


def pil_mask_to_numpy(mask: Image) -> np.ndarray:
    return (np.asarray(mask) / 255).astype(np.float32)


def numpy_mask_to_pil(mask: np.ndarray) -> Image:
    return Image.fromarray((mask * 255).astype(np.uint8)).convert("L")


def inpaint_tile(
    server: Inpainter | str,
    prompt: str,
    input_folder: str,
    input_image: str,
    output_image: Optional[str] = None,
    seed: int = 999,
    mode: Literal["single", "overlap-free"] = "single",
    extend_at: Optional[str] = None,
    ortho_scale: float = 1.75,
    base_ortho_scale: float = 1.75,
):
    if isinstance(server, str):
        inpainter = Inpainter(inpainter_type="flux_local", gradio_url=server)
    else:
        inpainter = server

    image_path = os.path.join(input_folder, input_image)
    mask_path = image_path.replace("rgb.png", "inpaint_mask.png")
    if output_image is None:
        output_path = image_path.replace("rgb.png", "inpainted.png")
    else:
        output_path = os.path.join(input_folder, output_image)

    base = Image.open(image_path).convert("RGB")
    mask = process_mask(Image.open(mask_path))

    MAX_OVERLAP = 4  # in pixels

    # depending on the ortho scale, we allow fewer pixels to overlap
    overlap = int((base_ortho_scale / ortho_scale) * MAX_OVERLAP)

    # we also erode the original mask a bit
    erosion_radius = int((base_ortho_scale / ortho_scale) * 4)

    overlap_mask = numpy_mask_to_pil(
        make_non_overlapping_mask(
            pil_mask_to_numpy(mask),
            extend_at=extend_at,
            full_mask=(mode == "single"),
            overlap_x=overlap,
            overlap_y=overlap // 2,
            erosion_radius=erosion_radius,
        )
    )
    overlap_mask.save(mask_path.replace("inpaint", "overlap-free"))
    numpy_mask_to_pil(
        make_non_overlapping_mask(
            pil_mask_to_numpy(mask),
            extend_at=None,
            overlap_x=0,
            overlap_y=0,
            add_corner=False,
            full_mask=(mode == "single"),
            erosion_radius=erosion_radius,
        )
    ).save(mask_path)
    image_inpainted = inpainter(base, overlap_mask, seed, prompt)
    image_inpainted.save(output_path)


def run_trellis(
    pipe,
    image_path,
    seed=1,
    mesh_path="./assets/house-tile.glb",
    metric_thresholds={"squareness": 1, "slab_size": 4096, "completeness": 0.95},
):
    from trellis.utils import render_utils, postprocessing_utils
    import torch

    image = Image.open(image_path)

    outputs = pipe.run(
        image,
        seed=seed,
        report_metrics=True,
        metric_thresholds=metric_thresholds,
    )

    for metric_name in ("squareness", "slab_size", "completeness"):
        print(f"{metric_name}: {outputs[metric_name]}")

    video = render_utils.render_video(outputs["scene"]["gaussian"][0])["color"]
    imageio.mimsave(image_path.replace(".png", ".mp4"), video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs["scene"]["gaussian"][0],
        outputs["scene"]["mesh"][0],
        # Optional parameters
        simplify=0.95,  # Ratio of triangles to remove in the simplification process
        texture_size=1024,  # Size of the texture used for the GLB
    )
    glb.export(mesh_path)

    del glb

    outputs_to_save = {k: v for k, v in outputs.items() if k not in ["scene"]}
    torch.save(outputs_to_save, mesh_path.replace(".glb", ".pt"))

    for k in [k for k in outputs["scene"].keys() if k != "gaussian"]:
        del outputs["scene"][k]

    return outputs


def get_widest_point_y(image, find_last=False):
    arr = np.array(image)

    if arr.ndim == 3:
        alpha = arr[..., -1]  # Get the alpha channel
    else:
        alpha = arr

    height, width = alpha.shape

    # indicating that the pixel is definitely not transparent
    mask = alpha > 200

    # check which rows contain at least one non-transparent pixel
    has_nonzero = mask.any(axis=1)
    if not np.any(has_nonzero):
        return None  # No non-transparent pixels found

    # initialize arrays for the first and last non-transparent x indices for each row
    first = np.zeros(height, dtype=int)
    last = np.zeros(height, dtype=int)

    # for rows that have non-transparent pixels, find the first occurrence along x
    first[has_nonzero] = np.argmax(mask[has_nonzero], axis=1)

    # for the last occurrence, reverse each row and use argmax, then adjust the index
    last[has_nonzero] = width - 1 - np.argmax(mask[has_nonzero, ::-1], axis=1)

    # compute the horizontal span (width) for each row
    spans = last - first

    # the row with the maximum span is our "widest point"
    if not find_last:
        widest_y = int(np.argmax(spans))
    else:
        # reverse the spans and find the last occurence of the maximum
        widest_y = int(height - 1 - np.argmax(spans[::-1]))

    return widest_y


def center_on_square(square_img, intricate_img):
    square_ground_y = get_widest_point_y(square_img)
    intricate_ground_y = get_widest_point_y(intricate_img)

    offset_y = square_ground_y - intricate_ground_y

    new_intricate = Image.new("RGBA", intricate_img.size, (0, 0, 0, 0))
    new_intricate.paste(intricate_img, (0, offset_y))

    return new_intricate


def rebased_inpainted_tile(
    inpainted_image_path,
    base_slab_path,
    is_left_tile: bool = True,
    scale=0.85,
    postfix="inpainted",
    erosion_radius=5,
    ortho_scale=1.75,
    base_ortho_scale=1.75,
    render_resolution=1024,
):
    if ortho_scale != base_ortho_scale:
        ortho_rescale = base_ortho_scale / ortho_scale
        crop_size = int(render_resolution * ortho_rescale)
        crop_size_sides = (render_resolution - crop_size) // 2

        # crop the images to the same size
        for fn in glob.glob(
            os.path.join(os.path.dirname(inpainted_image_path), "000_*.png")
        ):
            if "backup" in fn:
                continue

            # we are saving a backup of the original image
            shutil.copy(fn, fn.replace(".png", "_backup.png"))

            img = Image.open(fn)
            img = img.crop(
                (
                    crop_size_sides,
                    crop_size_sides,
                    crop_size_sides + crop_size,
                    crop_size_sides + crop_size,
                )
            )
            img.save(fn)

    inpainted_image = Image.open(inpainted_image_path)
    width, height = inpainted_image.size

    base_slab = Image.open(base_slab_path)

    # we need to rescale the base slab to the same size as the inpainted image
    base_slab = base_slab.resize((width, height))

    # mask out the slab
    conditioning_mask = Image.open(
        inpainted_image_path.replace(postfix, "conditioning_mask")
    ).convert("L")
    inpaint_mask = Image.open(
        inpainted_image_path.replace(postfix, "inpaint_mask")
    ).convert("L")

    discard_mask = PIL.ImageOps.invert(conditioning_mask)

    if is_left_tile:
        slab_mask = Image.composite(
            conditioning_mask,
            Image.new("L", conditioning_mask.size, (0,)),
            PIL.ImageOps.invert(inpaint_mask),
        )
        slabless = Image.composite(
            inpainted_image,
            Image.new("RGBA", inpainted_image.size, (0, 0, 0, 255)),
            PIL.ImageOps.invert(slab_mask),
        )
    else:
        slabless = Image.composite(
            inpainted_image,
            Image.new("RGBA", inpainted_image.size, (0, 0, 0, 255)),
            inpaint_mask,
        )

    slabless = Image.composite(
        slabless,
        Image.new("RGBA", inpainted_image.size, (0, 0, 0, 255)),
        PIL.ImageOps.invert(discard_mask),
    )

    # isolate the object
    isolated = rembg.remove(slabless, alpha_matting=True, post_process_mask=True)

    # erode the rembg result a bit in case there are is a white gradient around the object
    isolated_np = (np.asarray(isolated) / 255).astype(np.float32)
    isolated_np[..., -1] = np.where(isolated_np[..., -1] > 0.5, 1, 0)
    isolated_np[..., -1] = cv2.erode(
        isolated_np[..., -1], np.ones((erosion_radius, erosion_radius), np.uint8)
    )
    isolated = Image.fromarray((isolated_np * 255).astype(np.uint8))

    # make sure everything on the slab is actually retained
    # sometimes, rembg will remove some pixels on the slab which will break
    # the rebasing process
    slab_surface = Image.composite(
        base_slab.split()[-1], Image.new("L", base_slab.size, (0,)), inpaint_mask
    )

    # we'll erode this mask to make sure the object is not too close to the edge
    slab_surface_np = pil_mask_to_numpy(slab_surface)
    slab_surface_np = cv2.erode(
        slab_surface_np, np.ones((erosion_radius, erosion_radius), np.uint8)
    )
    slab_surface_eroded = numpy_mask_to_pil(slab_surface_np)

    isolated = Image.composite(inpainted_image, isolated, slab_surface_eroded)

    # scale the object
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    isolated = isolated.resize((scaled_width, scaled_height))

    # after resizing, the mask will not be perfectly binary (again)
    # thus, we have to binarize it again. to avoid any background ghosting,
    # we use a super strict threshold here, pulling down everything to 0 that's not 1
    isolated_np = (np.asarray(isolated) / 255).astype(np.float32)
    isolated_np[..., -1] = np.where(isolated_np[..., -1] == 1, 1, 0)
    isolated = Image.fromarray((isolated_np * 255).astype(np.uint8))

    # reposition the object so it is centered again after scaling
    repositioned = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    paste_x = (width - scaled_width) // 2
    paste_y = (height - scaled_height) // 2

    repositioned.paste(isolated, (paste_x, paste_y), mask=isolated)

    # rebase the object onto the original square, centering it
    merged = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    merged.paste(base_slab, (0, 0))

    centered_object = center_on_square(base_slab, repositioned)

    merged.paste(centered_object, (0, 0), mask=centered_object)

    return merged


def worker(
    prefix,
    tile_dict,
    inpainter_type,
    gradio_url,
    blender_path,
    gpu_queue,
    generated_grid,
    first_tile_path,
    tile_mq,
    task_id,
    config,
    init_seed=429,
    verbose=True,
):
    if not verbose:
        sys.stdout = open("/dev/null", "w")
        sys.stderr = open("/dev/null", "w")

    pos = (tile_dict["x"], tile_dict["y"])
    pos_str = f"{pos[0]},{pos[1]}"

    gpu_id = gpu_queue.get()
    tile_mq.put(
        {"pos": pos, "state": States.ASSIGNED, "task_id": task_id, "gpu_id": gpu_id}
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.trellis_image_to_3d import PoorTileQualityException
    from tile_cutting import find_cuts

    # create local copy of generated_grid
    generated_grid = json.loads(json.dumps(generated_grid))

    current_base_scale = (
        BASE_ORTHO_SCALE if (pos[0] == 0 and pos[1] == 0) else INITIAL_ORTHO_SCALE
    )
    current_ortho_scale = current_base_scale + ORTHO_SCALE_Y_STEP * min(
        pos[1], ORTHO_SCALE_MAX_NUM_Y_STEPS
    )

    tile_path = os.path.join(prefix, pos_str)

    os.makedirs(tile_path, exist_ok=True)

    has_loaded_pipeline = threading.Event()
    pipeline_queue = Queue()

    def load_pipeline_in_thread(event, queue):
        # Load the pipeline in a separate thread to avoid blocking the main thread
        thread_pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "JeffreyXiang/TRELLIS-image-large"
        )
        thread_pipeline.to("cuda")
        queue.put(thread_pipeline)
        event.set()

    pipeline_thread = threading.Thread(
        target=load_pipeline_in_thread, args=(has_loaded_pipeline, pipeline_queue)
    )
    pipeline_thread.start()

    pipeline = None

    attempts = 0
    while attempts < 10:
        # render again to restore the masks to their original state
        tile_mq.put({"pos": pos, "state": States.RENDERING, "task_id": task_id})

        render_next_tile(
            blender_path,
            grid=generated_grid,
            output_folder=tile_path,
            pos=pos,
            ortho_scale=current_ortho_scale,
        )

        mode = "single" if (pos[0] == 0 and pos[1] == 0) else "overlap-free"

        is_y_obstructed = any((t["y"] < pos[1] for t in generated_grid))

        if is_y_obstructed and pos[0] != 0:
            extend_at = "xy"
        elif is_y_obstructed:
            extend_at = "y"
        elif not (pos[0] == 0 and pos[1] == 0):
            extend_at = "x"
        else:
            extend_at = None

        seed = init_seed + task_id + attempts

        inpainting_server = Inpainter(inpainter_type, gradio_url)

        tile_mq.put({"pos": pos, "state": States.INPAINTING, "task_id": task_id})

        try:
            inpaint_tile(
                inpainting_server,
                tile_dict["prompt"],
                tile_path,
                "000_rgb.png",
                seed=seed,
                mode=mode,
                extend_at=extend_at,
                ortho_scale=current_ortho_scale,
                base_ortho_scale=current_base_scale,
            )

        except Exception as e:
            print(f"Encountered an error while inpainting tile {pos_str}: {e}")

            continue

        tile_mq.put({"pos": pos, "state": States.REBASING, "task_id": task_id})

        rebased = rebased_inpainted_tile(
            os.path.join(tile_path, "000_inpainted.png"),
            os.path.join(first_tile_path, "000_rgb.png"),
            is_left_tile=(pos[0] == 0),
            base_ortho_scale=BASE_ORTHO_SCALE,
            ortho_scale=current_ortho_scale,
            scale=config.rebasing_scale if hasattr(config, "rebasing_scale") else 0.85,
        )
        rebased.save(os.path.join(tile_path, "000_rebased.png"))

        tile_mesh_path = os.path.join(tile_path, "tile.glb")

        tile_mq.put({"pos": pos, "state": States.GENERATING, "task_id": task_id})

        try:
            if not has_loaded_pipeline.is_set():
                tile_mq.put({"pos": pos, "state": States.STALLED, "task_id": task_id})
                has_loaded_pipeline.wait()
                tile_mq.put(
                    {"pos": pos, "state": States.GENERATING, "task_id": task_id}
                )

            if pipeline is None:
                pipeline = pipeline_queue.get()

            pipeline_thread.join()

            outputs = run_trellis(
                pipeline,
                os.path.join(tile_path, "000_rebased.png"),
                mesh_path=tile_mesh_path,
            )
            gs = outputs["scene"]["gaussian"][0]
            break

        except PoorTileQualityException as e:
            print(e)
            attempts += 1

        except RuntimeError as e:
            # We have to restart the entire process if spconv crashes
            print(e)
            tile_mq.put({"pos": pos, "state": States.CRASH, "task_id": task_id})

            return

    # Clear memory after each iteration to avoid memory leaks
    # release the model
    for k in list(pipeline.models.keys()):
        del pipeline.models[k]

    del pipeline.sparse_structure_sampler
    del pipeline.slat_sampler

    del pipeline
    del pipeline_queue

    torch.cuda.empty_cache()

    tile_mq.put({"pos": pos, "state": States.CUTTING, "task_id": task_id})

    gs.save_ply(os.path.join(tile_path, "mesh.ply"), transform=None)
    del gs

    for k in list(outputs["scene"].keys()):
        del outputs["scene"][k]

    for k in list(outputs.keys()):
        del outputs[k]

    torch.cuda.empty_cache()

    tile_dict = {
        "path": tile_mesh_path,
        "x": pos[0],
        "y": pos[1],
        "seed": seed,
        **find_cuts(gaussian_path=os.path.join(tile_path, "mesh.ply")),
    }

    torch.cuda.empty_cache()

    tile_mq.put({"pos": pos, "state": States.ORIENTING, "task_id": task_id})

    tile_dict["rotation"] = find_orientation_of_tile(
        blender_path, tile_dict, os.path.join(tile_path, "000_rebased.png"), tile_path
    )

    with open(os.path.join(tile_path, "grid.json"), "w") as f:
        json.dump(generated_grid, f)

    generate_tile_info(blender_path, [tile_dict], tile_path)

    with open(os.path.join(tile_path, "tile_info.json"), "r") as f:
        tile_info = json.load(f)

    tile_dict = {**tile_info[0], **tile_dict}

    tile_mq.put({"pos": pos, "state": States.DONE, "task_id": task_id})

    torch.cuda.empty_cache()

    return tile_dict


def main(
    instructions: str = "demo.json",
    prefix: str = "run_new_prompts/loop",
    parallel: bool = True,
    workers: int = -1,
    gpu_ids: List[int] = None,
    skip_existing: bool = False,
    workers_per_gpu: int = 1,
    seed: int = 1429,
    gradio_url: str = "http://127.0.0.1:7860",
    inpainter_type: Literal[
        "flux_local", "flux_replicate", "sdxl_replicate"
    ] = "flux_local",
    blender_path: str = "blender-3.6.19-linux-x64/blender",
    resample: Tuple[int, int] = None,
    resample_prompt: str = None,
    **kwargs,
):

    assert (
        "CUDA_HOME" in os.environ
    ), "CUDA_HOME not set. Please restart the script prefixed with 'CUDA_HOME=/path/to/cuda'"

    os.makedirs(prefix, exist_ok=True)

    config = edict(kwargs)

    instructions = process_instructions(instructions)

    with open(os.path.join(prefix, "instructions.json"), "w") as f:
        json.dump(instructions["tiles"], f, indent=4)

    if resample is not None and isinstance(resample, str):
        resample = tuple(map(int, resample.split(",")))

    def has_instructions(x, y):
        return any(tile["x"] == x and tile["y"] == y for tile in instructions["tiles"])

    def unlock_adjacent_tiles(state_grid, pos):
        # unlock the next tile on the x axis (but only if we are not waiting for a tile below)
        if has_instructions(pos[0] + 1, pos[1]) and (
            (pos[1] - 1) < 0 or state_grid[pos[0] + 1][pos[1] - 1] == States.DONE
        ):
            if state_grid[pos[0] + 1][pos[1]] == States.BLOCKED:
                state_grid[pos[0] + 1][pos[1]] = States.READY

        # unlock the next tile on the y axis, but only if we are the first tile (or we were waiting)
        if has_instructions(pos[0], pos[1] + 1) and (
            pos[0] == 0 or state_grid[pos[0] - 1][pos[1] + 1] == States.DONE
        ):
            if state_grid[pos[0]][pos[1] + 1] == States.BLOCKED:
                state_grid[pos[0]][pos[1] + 1] = States.READY

    if skip_existing or resample is not None:
        # load the grid from disk
        with open(os.path.join(prefix, "grid.json"), "r") as f:
            generated_grid = json.load(f)

    else:
        generated_grid = []

    first_tile_path = os.path.join(prefix, "0,0")

    multiprocessing.set_start_method("forkserver", force=True)

    if gpu_ids is None:
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    gpu_ids = [int(gpu_id) for gpu_id in gpu_ids] * workers_per_gpu

    if parallel:
        gpu_ids = gpu_ids[:1]

    workers = len(gpu_ids) if workers == -1 else workers

    m = multiprocessing.Manager()
    gpu_queue = m.Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    gpu_logbook = {}

    tile_mq = m.Queue()

    if resample is not None:
        parallel = False

        for tile_instruction in instructions["tiles"]:
            if (
                tile_instruction["x"] == resample[0]
                and tile_instruction["y"] == resample[1]
            ):
                tile_instruction["prompt"] = instructions["prompt"].format(
                    **{"tile_prompt": resample_prompt}
                )

        generated_grid = [
            tile
            for tile in generated_grid
            if not (tile["x"] == resample[0] and tile["y"] == resample[1])
        ]

    if not parallel:
        # emit all messages
        old_tile_mq_put = tile_mq.put
        tile_mq.put = lambda x: [print(x), old_tile_mq_put(x)]

    state_grid = defaultdict(lambda: defaultdict(lambda: States.BLOCKED))
    for tile in instructions["tiles"]:
        state_grid[tile["x"]][tile["y"]] = States.BLOCKED

    if not skip_existing and resample is None:
        state_grid[0][0] = States.READY

    else:
        # manually unlock the next tiles
        for tile in generated_grid:
            state_grid[tile["x"]][tile["y"]] = States.DONE
            tile_mq.put(
                {"pos": (tile["x"], tile["y"]), "state": States.DONE, "task_id": -1}
            )
            unlock_adjacent_tiles(state_grid, (tile["x"], tile["y"]))

        if resample_prompt is not None:
            with open(os.path.join(prefix, "instructions.json"), "w") as f:
                json.dump(instructions["tiles"], f, indent=4)

    if parallel:
        pool = multiprocessing.Pool(processes=workers, maxtasksperchild=1)

    results = []

    def announce_crash(tile, task_id, e):
        tile_mq.put(
            {
                "pos": (tile["x"], tile["y"]),
                "state": States.CRASH,
                "task_id": task_id,
                "error": str(e),
            }
        )

    def queue_available_jobs():
        for tile in instructions["tiles"]:
            if state_grid[tile["x"]][tile["y"]] != States.READY:
                continue

            state_grid[tile["x"]][tile["y"]] = States.ASSIGNED

            task_id = len(results)

            if parallel:
                res = pool.apply_async(
                    worker,
                    (
                        prefix,
                        tile,
                        gradio_url,
                        blender_path,
                        gpu_queue,
                        generated_grid,
                        first_tile_path,
                        tile_mq,
                        task_id,
                        config,
                        seed,
                    ),
                    error_callback=partial(announce_crash, tile, task_id),
                )
            else:
                # peek at the gpu queue
                gpu_id = gpu_queue.get()
                gpu_queue.put(gpu_id)
                res = worker(
                    prefix,
                    tile,
                    inpainter_type,
                    gradio_url,
                    blender_path,
                    gpu_queue,
                    generated_grid,
                    first_tile_path,
                    tile_mq,
                    task_id,
                    config,
                    seed,
                    verbose=True,
                )
                gpu_queue.put(gpu_id)

            results.append(res)

    queue_available_jobs()

    while True:
        # we are done!
        if all(
            all(state == States.DONE for state in row.values())
            for row in state_grid.values()
        ):
            break

        message = tile_mq.get()

        print(message)

        pos = message["pos"]
        state = message["state"]
        task_id = message["task_id"]

        if "gpu_id" in message:
            gpu_logbook[task_id] = message["gpu_id"]

        state_grid[pos[0]][pos[1]] = state

        if state == States.DONE:
            if task_id != -1:
                generated_grid.append(
                    results[task_id].get() if parallel else results[task_id]
                )
                # release the gpu
                gpu_queue.put(gpu_logbook[task_id])
                del gpu_logbook[task_id]

            unlock_adjacent_tiles(state_grid, pos)

            # write the grid to disk
            with open(os.path.join(prefix, "grid.json"), "w") as f:
                json.dump(generated_grid, f, indent=4)

        elif state == States.CRASH:
            state_grid[pos[0]][pos[1]] = States.READY
            results[task_id] = None
            # release the gpu
            gpu_queue.put(gpu_logbook[task_id])
            del gpu_logbook[task_id]

        queue_available_jobs()

    if parallel:
        pool.close()
        pool.join()

    m.shutdown()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
