"""ARCHIVED: position-embedding and image-patchify paths of the old TransformerEncoder.

Removed during the 2026-10-28 refactor. No model dictionary selected
``position_embedding_type`` other than ``"none"`` and nothing set
``enable_patchify_input``; the ``rope``/``mrope`` branches were broken (they
passed an unsupported ``static=`` keyword) and the absolute-embedding branch read
the wrong dimension. Image patchification belongs to a model-specific input
adapter, not to the generic encoder. The full previous class is kept below for
reference only; it does not import.
"""

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
                self.cls_token = torch.nn.Parameter(torch.zeros(1, self.embed_dim))

        self.blocks = torch.nn.ModuleList(
            [TransformerEncoderBlock(config, block_index=i) for i in range(self.depth)]
        )

        if getattr(config, "use_final_norm", True):
            if config.norm_type == "rmsnorm":
                self.norm = RMSNorm(self.embed_dim, eps=getattr(config, "rms_norm_eps", 1e-5))
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
                    base=rope_theta_from_config(config),
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
                        base=rope_theta_from_config(config),
                        section_bases=mrope_section_bases,
                    )
                ]
            )
        elif self.position_embedding_type == "trained_absolute":
            max_pos = int(getattr(config, "max_position_embeddings", 2048))
            self.position_embeddings = torch.nn.Parameter(
                torch.zeros(max_pos, self.embed_dim)
            )
        elif self.position_embedding_type == "none":
            pass
        else:
            raise ValueError(
                f"Unsupported position_embedding_type: {self.position_embedding_type}"
            )

    def _image_to_patches(self, images):
        channels, height, width = images.shape
        if channels != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {channels} channels"
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "Input image size must be divisible by patch_size for patchify export path"
            )

        patches = images.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().permute(1, 2, 0, 3, 4)
        patches = patches.reshape(-1, channels * self.patch_size * self.patch_size)
        return patches

    def _prepare_input(self, x, preprocess_images=False):
        if preprocess_images:
            if x.ndim != 3:
                raise ValueError("With image preprocessing enabled, expected 3D image input")
            if not self.enable_patchify_input:
                raise ValueError(
                    "Image preprocessing requested but enable_patchify_input=False in config"
                )
            patches = self._image_to_patches(x)
            x = self.patch_projection(patches)
            if self.use_cls_token:
                x = torch.cat([self.cls_token, x], dim=0)
            return x

        if x.ndim != 2:
            raise ValueError("Expected token embedding input of shape (seq, hidden)")
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

        seq_len = x.size(0)
        if position_ids is None:
            positions = torch.arange(seq_len, device=x.device)
        else:
            if position_ids.ndim != 1:
                raise ValueError("Expected position_ids of shape (seq,)")
            positions = position_ids

        max_pos = self.position_embeddings.size(1)
        if int(positions.max()) >= max_pos:
            raise ValueError(
                f"position_ids exceed max_position_embeddings={max_pos}"
            )

        pos_emb = self.position_embeddings.index_select(0, positions.reshape(-1))
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

        rope = self._build_rope(seq_len=x.size(0), rope_pos=rope_pos)

        for block in self.blocks:
            x = block(x, rope=rope)

        x = self.norm(x)
        return x


__all__ = [
    "TransformerEncoderBlock",
    "TransformerEncoder",
]
