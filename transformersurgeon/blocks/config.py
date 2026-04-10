"""Model-style configuration for converted custom decoder graph compression."""

from ..utils.configuration import init_compressed_config


def build_custom_decoder_config(source_config_obj, indexing, compression_config=None):
    """Build a config instance using the same initialization pattern as model define files."""

    source_config_cls = source_config_obj.__class__

    class CustomDecoderConfigCompress(source_config_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            init_compressed_config(
                config_instance=self,
                indexing=indexing,
                compression_config=kwargs.get("compression_config", {}),
            )

    config_kwargs = source_config_obj.to_dict()
    config_kwargs["compression_config"] = compression_config or {}
    return CustomDecoderConfigCompress(**config_kwargs)


__all__ = ["build_custom_decoder_config"]
