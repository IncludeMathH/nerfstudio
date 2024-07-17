"""
use edge loss in NeRFacto
"""
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.renderers import EdgeRenderer
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, Type

from nerfstudio.model_components.losses import (
    MSELoss, 
    orientation_loss, 
    pred_normal_loss, 
    scale_gradients_by_distance_squared,
)

from nerfstudio.fields.nerfacto_edge_field import NerfactoEdgeField


@dataclass
class NerfactoEdgeModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: NerfactoEdgeModel)
    edge_loss_weight: float = 0.1

class NerfactoEdgeModel(NerfactoModel):
    config: NerfactoEdgeModelConfig

    def populate_modules(self):
        super().populate_modules()
        # Fields
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        appearance_embedding_dim = self.config.appearance_embed_dim if self.config.use_appearance_embedding else 0

        self.field = NerfactoEdgeField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            average_init_density=self.config.average_init_density,
            implementation=self.config.implementation,
        )
        self.edge_loss_weight = self.config.edge_loss_weight
        self.edge_loss = MSELoss()
        self.render_edge = EdgeRenderer(method='expected')

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # add edge prediction
        edge = self.render_edge(edge=field_outputs[FieldHeadNames.EDGE], weights=weights)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "edge": edge
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        pred_edge = outputs['edge']
        gt_edge = batch['edge'].to(self.device)
        loss_dict['edge_loss'] = self.compute_edge_loss(gt_edge, pred_edge)
        return loss_dict

    def compute_edge_loss(self, gt_edge, pred_edge):
        # compute edge loss
        return self.edge_loss(gt_edge, pred_edge) * self.edge_loss_weight
    
    def get_image_metrics_and_images(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        pred_edge = outputs['edge']                # (h, w)
        gt_edge = batch['edge'].to(self.device)    # (h, w, 1)
        combined_edge = torch.cat([gt_edge.unsqueeze(-1), pred_edge], dim=1)

        images_dict['edge'] = combined_edge
        return metrics_dict, images_dict
