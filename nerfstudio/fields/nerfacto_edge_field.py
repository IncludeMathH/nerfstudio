from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import get_normalized_directions
from nerfstudio.fields.nerfacto_field import NerfactoField
import torch
from torch import Tensor, nn
from typing import Dict, Optional, Literal
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.spatial_distortions import SpatialDistortion

class NerfactoEdgeField(NerfactoField):
    """Compound Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        base_res: base resolution of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        features_per_level: number of features per level for the hashgrid
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """
    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        average_init_density: float = 1.0,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ):
        super().__init__(
            aabb=aabb,
            num_images=num_images,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            geo_feat_dim=geo_feat_dim,
            num_levels=num_levels,
            base_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            num_layers_color=num_layers_color,
            num_layers_transient=num_layers_transient,
            features_per_level=features_per_level,
            hidden_dim_color=hidden_dim_color,
            hidden_dim_transient=hidden_dim_transient,
            appearance_embedding_dim=appearance_embedding_dim,
            transient_embedding_dim=transient_embedding_dim,
            use_transient_embedding=use_transient_embedding,
            use_semantics=use_semantics,
            num_semantic_classes=num_semantic_classes,
            pass_semantic_gradients=pass_semantic_gradients,
            use_pred_normals=use_pred_normals,
            use_average_appearance_embedding=use_average_appearance_embedding,
            spatial_distortion=spatial_distortion,
            average_init_density=average_init_density,
            implementation=implementation,
        )
        # Add your additional code here for the new layer
        self.mlp_edge_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=1,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        """
        density_embedding: 输出density是一同输出的特征
        """
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        embedded_appearance = None
        if self.embedding_appearance is not None:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
            ]
            + (
                [embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []
            ),
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        # add edge perdiction
        edge = self.mlp_edge_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.EDGE: edge})
        return outputs