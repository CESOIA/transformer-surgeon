def sort_bi_scores(final_bi):
    """
    final_bi is the output of eval_bi(model,...)
    {
        "vision": {
            0: {"layer_bi_score":..., "attn_bi_score":..., "mlp_bi_score":...},
            1: {...},
            ...
        },
        "text": {...}
    }
    """

    # -------------------------------
    # 1. Vision blockwise (layer_bi)
    # -------------------------------
    vision_block_sorted = sorted(
        [(layer, scores["layer_bi_score"]) 
         for layer, scores in final_bi["vision"].items()],
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------
    # 2. Text blockwise (layer_bi)
    # -------------------------------
    text_block_sorted = sorted(
        [(layer, scores["layer_bi_score"]) 
         for layer, scores in final_bi["text"].items()],
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------
    # 3. Vision MLP + Attention
    # -------------------------------
    vision_mlp_attn_sorted = []

    for layer, scores in final_bi["vision"].items():
        vision_mlp_attn_sorted.append((f"vision.{layer}.attn", scores["attn_bi_score"]))
        vision_mlp_attn_sorted.append((f"vision.{layer}.mlp",  scores["mlp_bi_score"]))

    vision_mlp_attn_sorted.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------
    # 4. Text MLP + Attention
    # -------------------------------
    text_mlp_attn_sorted = []

    for layer, scores in final_bi["text"].items():
        text_mlp_attn_sorted.append((f"text.{layer}.attn", scores["attn_bi_score"]))
        text_mlp_attn_sorted.append((f"text.{layer}.mlp",  scores["mlp_bi_score"]))

    text_mlp_attn_sorted.sort(key=lambda x: x[1], reverse=True)

    return dict(
        vision_block_sorted=vision_block_sorted,
        text_block_sorted=text_block_sorted,
        vision_mlp_attn_sorted=vision_mlp_attn_sorted,
        text_mlp_attn_sorted=text_mlp_attn_sorted,
    )
