from itertools import chain

import torch

from trellis.representations.gaussian import Gaussian
from trellis.renderers.sh_utils import SH2RGB

def collect_far_away_points(xyz, num=1000):
    """
    This function collects the num farthest points from each other.
    Returns a mask that can be used to filter the points.
    """
    # compute pairwise distances
    dist = torch.cdist(xyz, xyz)
    # get the indices of the num farthest points
    indices = torch.argsort(dist, descending=True, dim=1)[:, :num]
    # get the unique indices
    indices = torch.unique(indices.flatten())
    # create a mask
    mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    mask[indices] = True
    return mask

def compute_average_color(xyz, rgb):
    mask = collect_far_away_points(xyz, 100)

    return torch.mean(rgb[mask], dim=0)

def color_distance(c1, c2):
    # Euclidean distance in RGB space
    return torch.linalg.norm(c1 - c2)

def get_bin_edges(values, num_bins=50, mode='uniform'):
    """
    Compute bin edges along the given axis values.
    mode='uniform' uses linspace between min and max,
    mode='quantile' uses percentiles so that each bin has roughly equal count.
    """
    if mode == 'uniform':
        return torch.linspace(torch.min(values), torch.max(values), num_bins + 1, dtype=values.dtype, device=values.device)
    elif mode == 'quantile':
        percentiles = torch.linspace(0, 1, num_bins + 1, dtype=values.dtype, device=values.device)
        return torch.quantile(values, percentiles)
    else:
        raise ValueError("Mode must be 'uniform' or 'quantile'.")

def find_boundary(g, axis='x', side='min', num_bins=50, rgb_threshold=0.5, z_threshold=0.05, binning_mode='uniform', clean_tolerance=3.5e-2):
    """
    Scans from the outer edge inward along the specified axis ('x' or 'y')
    and returns the boundary where the average color or z value changes abruptly.
    """
    col = 0 if axis == 'x' else 1
    
    xyz = g.get_xyz
    shs = g.get_features
    rgb = SH2RGB(shs).clip(0, 1)

    valid_mask = xyz[:, -1] > xyz[:, -1].min() + 2e-2
    
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]
    
    values = xyz[:, col]
    bin_edges = get_bin_edges(values, num_bins=num_bins, mode=binning_mode)

    # Set up iteration order based on the side.
    if side == 'min':
        indices = range(num_bins)
    elif side == 'max':
        indices = range(num_bins-1, -1, -1)
    else:
        raise ValueError("Side must be 'min' or 'max'")
    
    # we collect all bin edges that we assume to be clean
    # which are values.min() + clean_tolerance or values.max() - clean_tolerance
    if side == 'min':
        clean_bin_edges = bin_edges < values.min() + clean_tolerance
    elif side == 'max':
        clean_bin_edges = bin_edges > values.max() - clean_tolerance
    
    max_z_value = -torch.inf
    for i in range(num_bins):
        if clean_bin_edges[i]:
            bin_mask = (values >= bin_edges[i]) & (values < bin_edges[i+1])
            max_z_bin = torch.max(xyz[bin_mask][:, -1])
            if max_z_bin > max_z_value:
                max_z_value = max_z_bin

    colors = []
    for i in indices:
        if clean_bin_edges[i]:
            bin_mask = (values >= bin_edges[i]) & (values < bin_edges[i+1])
            avg_color = compute_average_color(xyz[bin_mask], rgb[bin_mask])
            if not torch.isnan(avg_color).any():
                colors.append(avg_color)

    reference_color = torch.mean(torch.stack(colors), dim=0)
    
    colors = []
    for i in indices:
        bin_mask = (values >= bin_edges[i]) & (values < bin_edges[i+1])
        avg_color = compute_average_color(xyz[bin_mask], rgb[bin_mask])
        if torch.isnan(avg_color).any():
            continue
        d = color_distance(avg_color, reference_color)
        colors.append(avg_color)

    # calculate the maximum difference
    max_distance = torch.max(torch.stack([color_distance(c, reference_color) for c in colors]))
    rgb_threshold = max_distance * rgb_threshold

    rgb_boundary, z_boundary = None, None

    def has_all_boundaries():
        return rgb_boundary is not None and z_boundary is not None

    # Scan through bins from the edge inward.
    for i in indices:
        if clean_bin_edges[i]:
            continue

        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        if side == 'min':
            bin_mask = (values >= bin_start) & (values < bin_end)
        else:
            bin_mask = (values >= bin_start) & (values <= bin_end)
        bin_points = xyz[bin_mask]
        if len(bin_points) == 0:
            continue

        avg_color = compute_average_color(xyz[bin_mask], rgb[bin_mask])
        if torch.isnan(avg_color).any():
            continue

        d = color_distance(avg_color, reference_color)

        # If the color jump is larger than the threshold, we've hit the object edge.
        if rgb_boundary is None and d > rgb_threshold:
            # For 'min', return the start of the bin where the change occurred.
            # For 'max', return the end of that bin.
            rgb_boundary = bin_start if side == 'min' else bin_end

            # if the z boundary has been found, we enforce that this is the z boundary too if it hasn't been found yet
            if z_boundary is None:
                z_boundary = rgb_boundary

        # If the z value of the bin is higher than the current max_z_value, we've hit the object edge.
        max_z_bin = torch.max(bin_points[:, -1])
        if z_boundary is None and (max_z_bin - max_z_value) > z_threshold:
            z_boundary = bin_start if side == 'min' else bin_end
        
        if has_all_boundaries():
            break

    return rgb_boundary, z_boundary, max_z_value

def z_preserving_crop(self, x_range=None, y_range=None, z_range=None):
    """
    Crop Gaussian points to specified ranges in 3D space (in-place operation).
    
    Args:
        x_range (tuple, optional): (min_x, max_x) range to keep
        y_range (tuple, optional): (min_y, max_y) range to keep
        z_range (tuple, optional): (min_z, max_z) range to keep (globally)
        
    Returns:
        self: Returns self for method chaining
    """
    # Get world-space coordinates
    xyz = self.get_xyz
    
    # Initialize mask with all True
    mask = torch.ones(xyz.shape[0], dtype=torch.bool, device=self.device)
    
    # Apply range filters
    if x_range is not None:
        mask = mask & (xyz[:, 0] >= x_range[0]) & (xyz[:, 0] <= x_range[1])
    if y_range is not None:
        mask = mask & (xyz[:, 1] >= y_range[0]) & (xyz[:, 1] <= y_range[1])
    if z_range is not None:
        mask = mask | ((xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1]))
    # Apply mask to all attributes
    self._xyz = self._xyz[mask]
    self._features_dc = self._features_dc[mask]
    if self._features_rest is not None:
        self._features_rest = self._features_rest[mask]
    self._scaling = self._scaling[mask]
    self._rotation = self._rotation[mask]
    self._opacity = self._opacity[mask]
    
    return self

def ensure_squareness(min_rgb, max_rgb, min_z, max_z, mode="aggressive"):
    min_x_rgb, min_y_rgb = min_rgb
    max_x_rgb, max_y_rgb = max_rgb
    min_x_z, min_y_z = min_z
    max_x_z, max_y_z = max_z

    if mode == "aggressive":
        min_x_rgb, max_x_rgb = max(min_x_rgb, min_y_rgb), min(max_x_rgb, max_y_rgb)
        min_x_z, max_x_z = max(min_x_z, min_y_z), min(max_x_z, max_y_z)
    else:
        min_x_rgb, max_x_rgb = min(min_x_rgb, min_y_rgb), max(max_x_rgb, max_y_rgb)
        min_x_z, max_x_z = min(min_x_z, min_y_z), max(max_x_z, max_y_z)

    min_y_rgb, max_y_rgb = min_x_rgb, max_x_rgb
    min_y_z, max_y_z = min_x_z, max_x_z

    return (min_x_rgb, min_y_rgb), (max_x_rgb, max_y_rgb), (min_x_z, min_y_z), (max_x_z, max_y_z)

def find_cuts(gaussians=None, gaussian_path=None):
    if gaussian_path is not None:
        g = Gaussian(aabb=[-1, -1, -1, 1, 1, 1])
        g.load_ply(gaussian_path, transform=None)
    else:
        g = gaussians

    min_x_rgb, min_x_z, min_x_z_val = find_boundary(g, rgb_threshold=0.5, num_bins=64*8)
    max_x_rgb, max_x_z, max_x_z_val = find_boundary(g, side='max', rgb_threshold=0.5, num_bins=64*8)

    min_y_rgb, min_y_z, min_y_z_val = find_boundary(g, axis='y', rgb_threshold=0.5, num_bins=64*8)
    max_y_rgb, max_y_z, max_y_z_val = find_boundary(g, axis='y', side='max', rgb_threshold=0.5, num_bins=64*8)

    min_rgb, max_rgb, min_z, max_z = ensure_squareness(
        (min_x_rgb, min_y_rgb),
        (max_x_rgb, max_y_rgb),
        (min_x_z, min_y_z),
        (max_x_z, max_y_z),
        mode="aggressive"
    )

    if any([c < 0.2 for c in chain(min_rgb, max_rgb, min_z, max_z)]):
        # if any of the cuts are too close to the center, we don't trust them
        # and revert to the converative strategy
        min_rgb, max_rgb, min_z, max_z = ensure_squareness(
            (min_x_rgb, min_y_rgb),
            (max_x_rgb, max_y_rgb),
            (min_x_z, min_y_z),
            (max_x_z, max_y_z),
            mode="conservative"
        )

    min_x_rgb, min_y_rgb = min_rgb
    max_x_rgb, max_y_rgb = max_rgb

    min_x_z, min_y_z = min_z
    max_x_z, max_y_z = max_z

    avg_z_val = torch.stack([min_x_z_val, max_x_z_val, min_y_z_val, max_y_z_val]).mean(dim=0)

    # write a json file with the results, grouped by cut type (rgb or z)
    cuts = {
        "rgb_cut": {
            "min_x": min_x_rgb.item(),
            "max_x": max_x_rgb.item(),
            "min_y": min_y_rgb.item(),
            "max_y": max_y_rgb.item()
        },
        "z_preserving_cut": {
            "min_x": min_x_z.item(),
            "max_x": max_x_z.item(),
            "min_y": min_y_z.item(),
            "max_y": max_y_z.item(),
        },
        "avg_z": avg_z_val.item()
    }

    g_cropped = g.crop(x_range=(min_x_z, max_x_z), y_range=(min_y_z, max_y_z))
    g_cropped = z_preserving_crop(g_cropped, x_range=(min_x_rgb, max_x_rgb), y_range=(min_y_rgb, max_y_rgb), z_range=(avg_z_val, 0.5))

    from trellis.utils import render_utils
    import imageio
    video = render_utils.render_video(g, num_frames=128, resolution=512, r=2)['color']
    imageio.mimsave(gaussian_path.replace('.ply', '.mp4'), video, fps=30)

    return cuts
