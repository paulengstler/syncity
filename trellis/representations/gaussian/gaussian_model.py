import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import utils3d
from typing import Literal

class Gaussian(torch.nn.Module):
    def __init__(
            self, 
            aabb : list,
            sh_degree : int = 0,
            mininum_kernel_size : float = 0.0,
            scaling_bias : float = 0.01,
            opacity_bias : float = 0.1,
            scaling_activation : str = "exp",
            device='cuda'
        ):
        super(Gaussian, self).__init__()
        self.init_params = {
            'aabb': aabb,
            'sh_degree': sh_degree,
            'mininum_kernel_size': mininum_kernel_size,
            'scaling_bias': scaling_bias,
            'opacity_bias': opacity_bias,
            'scaling_activation': scaling_activation,
        }
        
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size 
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = lambda x: x + torch.log(-torch.expm1(-x))

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.scale_bias = self.inverse_scaling_activation(torch.tensor(self.scaling_bias)).to(self.device)
        self.rots_bias = torch.zeros((4), device=self.device)
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(torch.tensor(self.opacity_bias)).to(self.device)

    def register_gsplat_params(self, params):
        """
        Register parameters for optimization.
        """
        gsplat_mapping = {
            "means": "_xyz",
            "scales": "_scaling",
            "opacities": "_opacity",
            "quats": "_rotation",
            "sh0": "_features_dc",
        }

        for name, value in params.items():
            # delete the attribute if it exists
            if hasattr(self, gsplat_mapping[name]):
                delattr(self, gsplat_mapping[name])

            self.register_parameter(gsplat_mapping[name], value)

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size ** 2
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])
    
    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]
    
    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=2) if self._features_rest is not None else self._features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :])
    
    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias
        
    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]
    
    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        
    def from_features(self, features):
        self._features_dc = features
        
    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
        
    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()
        
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform.T)
            rotation = utils3d.numpy.quaternion_to_matrix(rotation)
            rotation = np.matmul(transform, rotation)
            rotation = utils3d.numpy.matrix_to_quaternion(rotation)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        if transform is not None:
            transform = np.array(transform)
            xyz = np.matmul(xyz, transform)
            rotation = utils3d.numpy.quaternion_to_matrix(rots)
            rotation = np.matmul(rotation, transform)
            rots = utils3d.numpy.matrix_to_quaternion(rotation)
            
        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            features_extra = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=self.device))
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)
        
        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = self.inverse_scaling_activation(torch.sqrt(torch.square(scales) - self.mininum_kernel_size ** 2)) - self.scale_bias
        self._rotation = rots - self.rots_bias[None, :]

    def crop(self, x_range=None, y_range=None, z_range=None):
        """
        Crop Gaussian points to specified ranges in 3D space (in-place operation).
        
        Args:
            x_range (tuple, optional): (min_x, max_x) range to keep
            y_range (tuple, optional): (min_y, max_y) range to keep
            z_range (tuple, optional): (min_z, max_z) range to keep
            
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
            mask = mask & (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])
        # Apply mask to all attributes
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        if self._features_rest is not None:
            self._features_rest = self._features_rest[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]
        
        return self
    
    def rotate(self, axis: Literal['X', 'Y', 'Z'], angle: float, degrees: bool = True):
        """
        Rotate Gaussian points around the specified axis by the specified angle (in-place operation).

        Args:
            axis (str): Axis to rotate around ('X', 'Y', or 'Z')
            angle (float): Angle to rotate by
            degrees (bool): If True, angle is in degrees; if False, angle is in radians

        Returns:
            self: Returns self for method chaining
        """

        rotation_matrix_np = utils3d.numpy.euler_axis_angle_rotation(axis.upper(), np.radians(angle) if degrees else angle)
        rotation_matrix = torch.tensor(rotation_matrix_np, dtype=torch.float32, device=self.device)

        self._xyz = ((self.get_xyz @ rotation_matrix.T) - self.aabb[None, :3]) / self.aabb[None, 3:]

        def quat_multiply(quaternion0, quaternion1):
            w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
            w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
            return torch.cat((
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            ), dim=-1)

        quaternions = utils3d.numpy.matrix_to_quaternion(rotation_matrix_np)[np.newaxis, ...]
        quaternions = torch.tensor(quaternions, dtype=torch.float32, device=self.device)
        rotations_from_quats = quat_multiply((self._rotation + self.rots_bias[None, :]), quaternions)
        normalized_rotations_from_quats = torch.linalg.norm(rotations_from_quats, dim=-1, keepdim=True)

        self._rotation = (rotations_from_quats / normalized_rotations_from_quats) - self.rots_bias[None, :]

        return self

    def translate(self, translation_xyz):
        """
        Translate Gaussian points by the specified amount in world space (in-place operation).
        
        Args:
            translation_xyz (list or tuple): [tx, ty, tz] translation amounts
            
        Returns:
            self: Returns self for method chaining
        """
        tx, ty, tz = translation_xyz
        translation = torch.tensor([tx, ty, tz], dtype=torch.float32, device=self.device)
        
        # Get world space coordinates, translate them, then convert back to normalized space
        xyz_world = self.get_xyz
        xyz_world_translated = xyz_world + translation
        
        # Convert back to normalized coordinates and update in-place
        self._xyz = (xyz_world_translated - self.aabb[None, :3]) / self.aabb[None, 3:]
        
        return self

    def translate_x(self, translation_x):
        return self.translate([translation_x, 0, 0])
    
    def translate_y(self, translation_y):
        return self.translate([0, translation_y, 0])
    
    def translate_z(self, translation_z):
        return self.translate([0, 0, translation_z])

    def copy(self):
        """
        Create a deep copy of this Gaussian object.
        
        Returns:
            Gaussian: A new Gaussian object with copies of all attributes
        """
        new_gaussian = Gaussian(
            aabb=self.aabb.tolist(),
            sh_degree=self.sh_degree,
            mininum_kernel_size=self.mininum_kernel_size,
            scaling_bias=self.scaling_bias,
            opacity_bias=self.opacity_bias,
            scaling_activation=self.scaling_activation_type,
            device=self.device
        )
        
        # Copy all attributes
        new_gaussian._xyz = self._xyz.clone()
        new_gaussian._features_dc = self._features_dc.clone()
        if self._features_rest is not None:
            new_gaussian._features_rest = self._features_rest.clone()
        new_gaussian._scaling = self._scaling.clone()
        new_gaussian._rotation = self._rotation.clone()
        new_gaussian._opacity = self._opacity.clone()
        
        return new_gaussian
    
    
    def change_scale(self, ratio, center=None):
        """
        Scale the Gaussian object by the specified ratio (in-place operation).
        
        This scales both the positions and the size of the Gaussians.
        
        Args:
            ratio (float): Scale factor to apply (e.g., 0.5 for half size)
                
        Returns:
            self: Returns self for method chaining
        """
        if not isinstance(ratio, (int, float)):
            raise TypeError(f"Scale ratio must be a number, got {type(ratio)}")
        
        if ratio <= 0:
            raise ValueError(f"Scale ratio must be positive, got {ratio}")
        
        # Get world space coordinates
        xyz_world = self.get_xyz
        
        # Calculate the center of the point cloud
        center = xyz_world.mean(dim=0) if center is None else torch.tensor(center, dtype=torch.float32, device=self.device)
        
        # Scale positions relative to center
        xyz_world_scaled = center + (xyz_world - center) * ratio
        
        # Convert back to normalized coordinates
        self._xyz = (xyz_world_scaled - self.aabb[None, :3]) / self.aabb[None, 3:]
        
        # Scale the Gaussian kernels (scaling parameters)
        # Note: we apply the ratio directly to the activated scales
        scales = self.get_scaling
        scaled_scales = scales * ratio
        self.from_scaling(scaled_scales)
        
        return self
    
    def combine(self, other_gaussians):
        """
        Combine other Gaussian objects into this one (in-place operation).
        
        Args:
            other_gaussians (list): List of other Gaussian objects to combine with this one
                
        Returns:
            self: Returns self for method chaining
        """
        if not other_gaussians:
            return self  # Nothing to combine
        
        # Prepare lists to collect all attributes
        all_xyz = [self._xyz]
        all_features_dc = [self._features_dc]
        all_features_rest = [] if self._features_rest is None else [self._features_rest]
        all_scaling = [self._scaling]
        all_rotation = [self._rotation]
        all_opacity = [self._opacity]
        
        # First check if any Gaussian has features_rest
        has_features_rest = self._features_rest is not None or any(g._features_rest is not None for g in other_gaussians)
        
        # Collect attributes from all other Gaussians
        for g in other_gaussians:
            # For each Gaussian, get world space xyz
            xyz_world = g.get_xyz
            
            # Convert to this Gaussian's normalized space
            xyz_normalized = (xyz_world - self.aabb[None, :3]) / self.aabb[None, 3:]
            
            all_xyz.append(xyz_normalized)
            all_features_dc.append(g._features_dc)
            
            # Handle features_rest (some might have it, some might not)
            if has_features_rest:
                if g._features_rest is not None:
                    all_features_rest.append(g._features_rest)
                else:
                    # Create zero padding with same shape as features_dc but matching the rest shape
                    rest_channels = self._features_rest.shape[2] if self._features_rest is not None else \
                                next((g._features_rest.shape[2] for g in other_gaussians if g._features_rest is not None), 0)
                    padding = torch.zeros(
                        (g._features_dc.shape[0], g._features_dc.shape[1], rest_channels),
                        dtype=g._features_dc.dtype,
                        device=g.device
                    )
                    all_features_rest.append(padding)
                    
            all_scaling.append(g._scaling)
            all_rotation.append(g._rotation)
            all_opacity.append(g._opacity)
        
        # Concatenate all collected attributes
        self._xyz = torch.cat(all_xyz, dim=0)
        self._features_dc = torch.cat(all_features_dc, dim=0)
        
        if has_features_rest:
            if not all_features_rest:  # If self._features_rest was None and no other gaussian had features_rest
                self._features_rest = None
            else:
                self._features_rest = torch.cat(all_features_rest, dim=0)
        else:
            self._features_rest = None
            
        self._scaling = torch.cat(all_scaling, dim=0)
        self._rotation = torch.cat(all_rotation, dim=0)
        self._opacity = torch.cat(all_opacity, dim=0)
        
        return self
        