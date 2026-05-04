import torch

from .atomic import AtomicSum
from .mha import MHAEncoder, MHAEncoderFusedProj
from .mlp import MLP, MLPGated
from .norm import RMSNorm
from .rope import (
    precompute_mrope_cos_sin_half,
    precompute_mrope_inv_freqs,
    precompute_rope_cos_sin_half,
    precompute_rope_inv_freqs,
)


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, config, block_index):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.mlp_hidden_dim = config.intermediate_size
        self.mlp_activation = config.hidden_act
        self.mha_type = config.attn_type
        self.mlp_type = config.mlp_type
        self.norm_type = config.norm_type
        self.norm_position = getattr(config, "norm_position", "pre")
        self.use_sdpa = config.use_sdpa

        self.kv_num_heads = getattr(config, "num_key_value_heads", None)

        self.compression_config = {
            "attn": {},
            "mlp": {},
        }
        prefix = f"blocks.{block_index}."
        for full_path, value in getattr(config, "compression_config", {}).items():
            if not full_path.startswith(prefix):
                continue
            local_path = full_path[len(prefix):]
            if local_path.startswith("attn."):
                self.compression_config["attn"][local_path.split(".", 1)[1]] = value
            elif local_path.startswith("mlp."):
                self.compression_config["mlp"][local_path.split(".", 1)[1]] = value

        self.bias_required = getattr(config, "bias_required", {"attn": {}, "mlp": {}})

        if self.norm_type == "rmsnorm":
            self.norm_in = RMSNorm(self.embed_dim)
            self.norm_out = RMSNorm(self.embed_dim)
        elif self.norm_type == "layernorm":
            self.norm_in = torch.nn.LayerNorm(self.embed_dim)
            self.norm_out = torch.nn.LayerNorm(self.embed_dim)
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")

        if self.mha_type == "mha_encoder":
            self.attn = MHAEncoder(
                self.embed_dim,
                self.num_heads,
                bias_required=self.bias_required["attn"],
                kv_num_heads=self.kv_num_heads,
                compression_config=self.compression_config["attn"],
                use_sdpa=self.use_sdpa,
            )
        elif self.mha_type == "mha_encoder_fused_proj":
            self.attn = MHAEncoderFusedProj(
                self.embed_dim,
                self.num_heads,
                bias_required=self.bias_required["attn"],
                compression_config=self.compression_config["attn"],
                use_sdpa=self.use_sdpa,
            )
        else:
            raise ValueError(f"Unsupported MHA type: {self.mha_type}")

        if self.mlp_type == "mlp_gated":
            self.mlp = MLPGated(
                self.embed_dim,
                self.mlp_hidden_dim,
                bias_required=self.bias_required["mlp"],
                activation=self.mlp_activation,
                compression_config=self.compression_config["mlp"],
            )
        elif self.mlp_type == "mlp":
            self.mlp = MLP(
                self.embed_dim,
                self.mlp_hidden_dim,
                bias_required=self.bias_required["mlp"],
                activation=self.mlp_activation,
                compression_config=self.compression_config["mlp"],
            )
        else:
            raise ValueError(f"Unsupported MLP type: {self.mlp_type}")

        self.atomic_sum = AtomicSum()

    def forward(self, x, rope=None):
        if self.norm_position == "pre":
            residual = x
            x = self.norm_in(x)
            x = self.attn(x, rope=rope)
            x = self.atomic_sum(x, residual)

            residual = x
            x = self.norm_out(x)
            x = self.mlp(x)
            x = self.atomic_sum(x, residual)
        elif self.norm_position == "post":
            residual = x
            x = self.attn(x, rope=rope)
            x = self.atomic_sum(x, residual)
            x = self.norm_in(x)

            residual = x
            x = self.mlp(x)
            x = self.atomic_sum(x, residual)
            x = self.norm_out(x)
        else:
            raise ValueError(f"Unsupported norm_position: {self.norm_position}")

        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.depth = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.position_embedding_type = getattr(config, "position_embedding_type", "none")

        self.enable_patchify_input = bool(getattr(config, "enable_patchify_input", False))
        self.patch_size = int(getattr(config, "patch_size", 16))
        self.num_channels = int(getattr(config, "num_channels", 3))
        self.use_cls_token = bool(getattr(config, "use_cls_token", False))

        if self.enable_patchify_input:
            patch_dim = self.num_channels * self.patch_size * self.patch_size
            self.patch_projection = torch.nn.Linear(
                patch_dim,
                self.embed_dim,
                bias=bool(getattr(config, "patch_bias", True)),
            )
            if self.use_cls_token:
                self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(config, block_index=i) for i in range(self.depth)]
        )

        if getattr(config, "use_final_norm", True):
            if config.norm_type == "rmsnorm":
                self.norm = RMSNorm(self.embed_dim)
            elif config.norm_type == "layernorm":
                self.norm = torch.nn.LayerNorm(self.embed_dim)
            else:
                raise ValueError(f"Unsupported norm type: {config.norm_type}")
        else:
            self.norm = torch.nn.Identity()

        head_dim = self.embed_dim // self.num_heads

        if self.position_embedding_type == "rope":
            self.inv_freq = torch.nn.Parameter(
                precompute_rope_inv_freqs(
                    head_dim=head_dim,
                    base=float(getattr(config, "rope_theta", 1e6)),
                ),
                requires_grad=False,
            )
        elif self.position_embedding_type == "mrope":
            mrope_section_dims = getattr(config, "mrope_section_dims", None)
            mrope_section_bases = getattr(config, "mrope_section_bases", None)
            self.inv_freq_sections = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(freq, requires_grad=False)
                    for freq in precompute_mrope_inv_freqs(
                        head_dim=head_dim,
                        section_dims=mrope_section_dims,
                        base=float(getattr(config, "rope_theta", 1e6)),
                        section_bases=mrope_section_bases,
                    )
                ]
            )
        elif self.position_embedding_type == "trained_absolute":
            max_pos = int(getattr(config, "max_position_embeddings", 2048))
            self.position_embeddings = torch.nn.Parameter(
                torch.zeros(1, max_pos, self.embed_dim)
            )
        elif self.position_embedding_type == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported position_embedding_type: {self.position_embedding_type}"
            )

    def _image_to_patches(self, images):
        batch_size, channels, height, width = images.shape
        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {channels} channels"
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "Input image size must be divisible by patch_size for patchify export path"
            )

        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(batch_size, -1, channels * self.patch_size * self.patch_size)
        return patches

    def _prepare_input(self, x, preprocess_images=False):
        if preprocess_images:
            if x.ndim != 4:
                raise ValueError("With image preprocessing enabled, expected 4D image input")
            if not self.enable_patchify_input:
                raise ValueError(
                    "Image preprocessing requested but enable_patchify_input=False in config"
                )
            patches = self._image_to_patches(x)
            x = self.patch_projection(patches)
            if self.use_cls_token:
                cls = self.cls_token.expand(x.size(0), -1, -1)
                x = torch.cat([cls, x], dim=1)
            return x

        if x.ndim != 3:
            raise ValueError("Expected token embedding input of shape (batch, seq, hidden)")
        return x

    def _build_rope(self, seq_len, rope_pos=0):
        if self.position_embedding_type == "rope":
            return precompute_rope_cos_sin_half(
                self.inv_freq,
                seq_len,
                rope_pos,
                static=False,
            )

        if self.position_embedding_type == "mrope":
            section_positions = getattr(self.config, "mrope_positions", None)
            if section_positions is None:
                section_positions = rope_pos
            return precompute_mrope_cos_sin_half(
                self.inv_freq_sections,
                seq_len,
                section_positions,
                static=False,
            )

        return None

    def _apply_absolute_positional_embeddings(self, x, position_ids=None):
        if self.position_embedding_type != "trained_absolute":
            return x

        seq_len = x.size(1)
        if position_ids is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        else:
            positions = position_ids

        max_pos = self.position_embeddings.size(1)
        if int(positions.max()) >= max_pos:
            raise ValueError(
                f"position_ids exceed max_position_embeddings={max_pos}"
            )

        pos_emb = self.position_embeddings.index_select(1, positions.reshape(-1))
        pos_emb = pos_emb.view(positions.size(0), positions.size(1), -1)
        return x + pos_emb

    def forward(
            self,
            x: torch.Tensor,
            preprocess_images: bool = False,
            position_ids: torch.Tensor = None,
            rope_pos: int = 0,
    ):
        x = self._prepare_input(
            x,
            preprocess_images=preprocess_images,
        )
        x = self._apply_absolute_positional_embeddings(x, position_ids=position_ids)

        rope = self._build_rope(seq_len=x.size(1), rope_pos=rope_pos)

        for block in self.blocks:
            x = block(x, rope=rope)

        x = self.norm(x)
        return x


__all__ = [
    "TransformerEncoderBlock",
    "TransformerEncoder",
]
