from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult


class PoorTileQualityException(Exception):
    pass

def count_contact_on_slice(voxel_grid, bounding_box, z_slice):
    """
    For a given voxel_grid and its bounding_box, count the number of voxels
    in the x-y slice at z=z_slice that lie on the boundary of the bounding box.
    """
    x_min, x_max, y_min, y_max, _, _ = bounding_box
    # Get the slice at z=z_slice (shape: 64 x 64)
    slice_voxels = (voxel_grid[:, :, z_slice] > 0).nonzero(as_tuple=False)  # (num_voxels, 2)
    if slice_voxels.numel() == 0:
        return 0
    # A voxel is in contact if its x coordinate is x_min or x_max or its y coordinate is y_min or y_max.
    contact_mask = ((slice_voxels[:, 0] == x_min) | (slice_voxels[:, 0] == x_max) |
                    (slice_voxels[:, 1] == y_min) | (slice_voxels[:, 1] == y_max))
    contact_count = int(contact_mask.sum().item())
    return contact_count

def choose_best_z_slice(voxel_grid, bounding_box):
    """
    Find the z slice within the object's bounding box that has the maximum number of voxels.
    Returns the z index and the number of voxels in that slice.
    """
    _, _, _, _, z_min, z_max = bounding_box
    best_z = None
    max_voxels = 0
    for z in range(z_min, z_max + 1):
        slice_voxels = (voxel_grid[:, :, z] > 0).nonzero(as_tuple=False)
        num_voxels = slice_voxels.size(0)
        if num_voxels > max_voxels:
            max_voxels = num_voxels
            best_z = z
    return best_z, max_voxels

def process_voxel_tensor(a, bounds):
    """
    Process a binary voxel tensor of shape (B, 64, 64, 64). For each object,
    compute the bounding box, pick the best z slice (the one with maximum occupancy),
    and count the number of voxels in that slice that contact the bounding box boundary.
    """
    B = a.shape[0]
    results = []
    for i in range(B):
        voxel_grid = a[i]
        bbox = bounds.view(B, -1)[i].int().tolist()
        # Choose the z slice with the maximum occupancy.
        best_z, _ = choose_best_z_slice(voxel_grid, bbox)
        contact_count = count_contact_on_slice(voxel_grid, bbox, best_z)
        results.append({
            'bounding_box': bbox,
            'z_slice': best_z,
            'contact_count': contact_count
        })
    return results

class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image, center_input: bool) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            #output = rembg.remove(input, session=self.rembg_session)
            output = np.concatenate([np.array(input),  (np.array(input).sum(-1, keepdims=True) > 30) * 255], -1)
            output = Image.fromarray(output.astype(np.uint8), 'RGBA')
            # TODO: FIX THIS!!!!!!! ^
        output_np = np.array(output)
        alpha = output_np[:, :, 3]

        if center_input:
            bbox = np.argwhere(alpha > 0.8 * 255)
            bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
            center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            size = int(size * 1.2)
            bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
            output = output.crop(bbox)  # type: ignore

        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        cond_samples: Optional[torch.Tensor] = None,
        cond_samples_mask: Optional[torch.Tensor] = None,
        cond_samples_decoded: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            cond_samples=cond_samples,
            cond_samples_mask=cond_samples_mask,
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        # Unclear if this code is needed. It was buggy without this but now it is OK -- keep just in case.
        # if cond_samples_decoded is not None:
        #     upsampled_mask = F.interpolate(
        #         cond_samples_mask, 
        #         size=(64, 64, 64), 
        #         mode='nearest'
        #     )
        #     coords_new = []
        #     shift = torch.tensor([0, 32, 0, 0]).to(coords.device).to(dtype=torch.int)
        #     for coord in cond_samples_decoded:
        #         if coord[1] >= 32:
        #             coord_shifted = coord - shift
        #             coords_new.append(coord_shifted)
        #     for coord in coords:
        #         if upsampled_mask[0, 0, coord[1], coord[2], coord[3]] > 0:
        #             coords_new.append(coord)

        #     coords = torch.stack(coords_new).to(coords.device)    
               

        return coords, z_s

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        cond_samples: Optional[torch.Tensor] = None,
        cond_samples_mask: Optional[torch.Tensor] = None,
    ) -> tuple[sp.SparseTensor, sp.SparseTensor]:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            cond_samples=cond_samples,
            cond_samples_mask=cond_samples_mask,
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat_norm = slat * std + mean
        
        return slat_norm, slat
    
    def run_metrics(self, coords) -> dict:
        ss = torch.zeros(len(coords[:, 0].unique(sorted=False)), 64, 64, 64, dtype=torch.long)
        ss[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]] = 1

        # Split by batch_indices
        batch_indices = coords[:, 0]
        point_indices = coords[:, 1:]
        
        # Split by batch_indices
        splits = torch.bincount(batch_indices)
        points_by_batch = torch.split(point_indices, splits.tolist())
        
        # Calculate min/max for each sample
        bounds = torch.stack([
            torch.stack([
                batch.amin(dim=0),
                batch.amax(dim=0)
            ], dim=1)
            for batch in points_by_batch
        ]) # (batch, 3, 2)

        extents = (bounds[:, :, 1] - bounds[:, :, 0] + 1).float() # (batch, 3)

        # 1) measure the "size" of the structure.
        slab_size = extents[:, :2].prod(dim=-1).mean()

        # 2) measure the "squareness" of the structure. it will be 1 for a perfect square and < 1 for a rectangle
        squareness: torch.Tensor = (min(extents[:, 0], extents[:, 1]) / max(extents[:, 0], extents[:, 1])).mean()

        border_contact = torch.tensor([r["contact_count"] for r in process_voxel_tensor(ss, bounds)])

        # make it relative to the bounding box size
        border_contact = border_contact.float().to(extents.device) / (extents[:, :2].sum(dim=-1) * 2 - 4)

        # 3) measure the "completeness" of the structure. it will be 1 if the structure has one solid xz border
        completeness = border_contact.mean()

        return {
            'slab_size': slab_size.long().item(),
            'squareness': squareness.item(),
            'completeness': completeness.item(),
        }

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        cond_ss: Optional[dict] = None,
        cond_slat: Optional[dict] = None,
        center_input: bool = True,
        # 0 means no thresholding
        metric_thresholds: Optional[dict] = None,
        report_metrics: bool = False,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image, center_input)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        cond_coords = cond_ss['latents'] if cond_ss is not None else None
        cond_coords_mask = cond_ss['mask'] if cond_ss is not None else None
        cond_coords_decoded = cond_ss['decoded'] if cond_ss is not None else None
        coords, coords_raw = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params, cond_coords, cond_coords_mask, cond_coords_decoded)

        if metric_thresholds is not None or report_metrics:
            metrics = self.run_metrics(coords)

        if metric_thresholds is not None:
            for k, v in metrics.items():
                # any metric below threshold will raise an exception
                if k in metric_thresholds and v < metric_thresholds[k]:
                    raise PoorTileQualityException(f"Metric {k} is below threshold: {v} < {metric_thresholds[k]}")

        if cond_slat is not None:
            cond_slat_filtered, cond_slat_mask_filtered = match_slat_by_coords(cond_slat['latents'], cond_slat['mask'], coords)
        else:
            cond_slat_filtered, cond_slat_mask_filtered = None, None

        slat, slat_raw = self.sample_slat(cond, coords, slat_sampler_params, cond_slat_filtered, cond_slat_mask_filtered)

        out_dict = {
            'scene': self.decode_slat(slat, formats), 
            'ss_latents' : coords_raw,
            'ss' : coords,
            'slat_latents' : slat_raw,
            'slat' : slat,
        }

        if report_metrics:
            out_dict = {**out_dict, **metrics}

        return out_dict

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode =='multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    def sample_slat_multi_image(
            self,
            coords,
            cond: dict = {},
            slat_sampler_params: dict = {},
            mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
            cond_samples: Optional[torch.Tensor] = None,
            cond_samples_mask: Optional[torch.Tensor] = None,

    ):
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', cond['cond'].shape[0], slat_steps, mode=mode):
            slat, slat_raw = self.sample_slat(cond, coords, slat_sampler_params, cond_samples=cond_samples, cond_samples_mask=cond_samples_mask)
        return slat, slat_raw
        
def match_slat_by_coords(cond_slat, cond_slat_mask, coords):
    # Some matching such that conditioning on the slats works as expected

    device = cond_slat.feats.device
    
    # Create a mapping from `cond_slat.coords` to `cond_slat.feats` and `cond_slat_mask.feats`
    slat_coord_map = {tuple(c.cpu().numpy()): i for i, c in enumerate(cond_slat.coords)}
    
    # Initialize new feats for cond_slat and cond_slat_mask
    new_feats = torch.zeros((coords.shape[0], cond_slat.feats.shape[1]), device=device)
    new_mask_feats = torch.ones((coords.shape[0], cond_slat_mask.feats.shape[1]), device=device)
    
    # Populate new feats based on matching coords
    for i, coord in enumerate(coords):
        coord_tuple = tuple(coord.cpu().numpy())
        if coord_tuple in slat_coord_map:
            idx = slat_coord_map[coord_tuple]
            if (cond_slat_mask.feats[idx] == 0).all():  # If the mask is all zeros
                new_feats[i] = cond_slat.feats[idx]
                new_mask_feats[i] = 0  # Mask remains zeros for used entries

    # Create new SparseTensors
    new_cond_slat = sp.SparseTensor(
        feats=new_feats,
        coords=coords
    )
    new_cond_slat_mask = sp.SparseTensor(
        feats=new_mask_feats,
        coords=coords
    )
    
    return new_cond_slat, new_cond_slat_mask