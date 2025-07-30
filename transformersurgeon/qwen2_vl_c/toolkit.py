from ..utils import CompressionScheme

def generate_compression_schemes(config):
    VISION_BLOCKS_NUM = 32
    TEXT_BLOCKS_NUM = 28

    vision_path_dicts = []
    for i in range(VISION_BLOCKS_NUM):
        vision_path_dicts.append(
            {
                "sa_qkv": CompressionScheme(
                    path=f"model.visual.blocks.{i}.attn.qkv",
                    pruning_ratio=config.vision_config.pruning_ratio_lists["sa_qkv"][i],
                    lrd_rank=config.vision_config.lrd_rank_lists["sa_qkv"][i],
                    is_qkv=True,
                ),
                "sa_out": CompressionScheme(
                    path=f"model.visual.blocks.{i}.attn.proj",
                    pruning_ratio=config.vision_config.pruning_ratio_skip_connections,
                    lrd_rank=config.vision_config.lrd_rank_lists["sa_out"][i],
                ),
                "mlp_up": CompressionScheme(
                    path=f"model.visual.blocks.{i}.mlp.fc1",
                    pruning_ratio=config.vision_config.pruning_ratio_lists["mlp_up"][i],
                    lrd_rank=config.vision_config.lrd_rank_lists["mlp_up"][i],
                ),
                "mlp_down": CompressionScheme(
                    path=f"model.visual.blocks.{i}.mlp.fc2",
                    pruning_ratio=config.vision_config.pruning_ratio_skip_connections,
                    lrd_rank=config.vision_config.lrd_rank_lists["mlp_down"][i],
                ),
            }
        )

    text_path_dicts = []
    for i in range(TEXT_BLOCKS_NUM):
        text_path_dicts.append(
            {
                "sa_q": CompressionScheme(
                    path=f"model.language_model.layers.{i}.self_attn.q_proj",
                    pruning_ratio=config.text_config.pruning_ratio_lists["sa_qkv"][i],
                    lrd_rank=config.text_config.lrd_rank_lists["sa_q"][i],
                ),
                "sa_k": CompressionScheme(
                    path=f"model.language_model.layers.{i}.self_attn.k_proj",
                    pruning_ratio=config.text_config.pruning_ratio_lists["sa_qkv"][i],
                    lrd_rank=config.text_config.lrd_rank_lists["sa_k"][i],
                ),
                "sa_v": CompressionScheme(
                    path=f"model.language_model.layers.{i}.self_attn.v_proj",
                    pruning_ratio=config.text_config.pruning_ratio_lists["sa_qkv"][i],
                    lrd_rank=config.text_config.lrd_rank_lists["sa_v"][i],
                ),
                "sa_out": CompressionScheme(
                    path=f"model.language_model.layers.{i}.self_attn.o_proj",
                    pruning_ratio=config.text_config.pruning_ratio_skip_connections,
                    lrd_rank=config.text_config.lrd_rank_lists["sa_out"][i],
                ),
                "mlp_gate": CompressionScheme(
                    path=f"model.language_model.layers.{i}.mlp.gate_proj",
                    pruning_ratio=config.text_config.pruning_ratio_lists["mlp_gate"][i],
                    lrd_rank=config.text_config.lrd_rank_lists["mlp_gate"][i],
                ),
                "mlp_up": CompressionScheme(
                    path=f"model.language_model.layers.{i}.mlp.up_proj",
                    pruning_ratio=config.text_config.pruning_ratio_lists["mlp_up"][i],
                    lrd_rank=config.text_config.lrd_rank_lists["mlp_up"][i],
                ),
                "mlp_down": CompressionScheme(
                    path=f"model.language_model.layers.{i}.mlp.down_proj",
                    pruning_ratio=config.text_config.pruning_ratio_skip_connections,
                    lrd_rank=config.text_config.lrd_rank_lists["mlp_down"][i],
                ),
            }
        )

    return {
        "vision": vision_path_dicts,
        "text": text_path_dicts,
    }
