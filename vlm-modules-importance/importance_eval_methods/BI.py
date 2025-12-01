import torch
from torch.nn import CosineSimilarity


def compute_bi(x_i, z_i, y_i):
    cos = CosineSimilarity(dim=-1)
    layer_bi = 1 - abs(cos(x_i, y_i).mean().item())
    attn_bi  = 1 - abs(cos(x_i, z_i).mean().item())
    mlp_bi   = 1 - abs(cos(z_i, y_i).mean().item())
    return layer_bi, attn_bi, mlp_bi


def get_module_by_name(model, full_path: str):
    parts = full_path.split('.')
    obj = model
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def register_hooks(model, indexing):
    
    block_inp = {mod: {} for mod in indexing}
    block_out = {mod: {} for mod in indexing}
    post_norm = {mod: {} for mod in indexing}

    def make_hook(mod_name, layer_idx, path):
        def hook_fn(module, input, output):
            block_inp[mod_name].setdefault(layer_idx, [])
            block_out[mod_name].setdefault(layer_idx, [])
            post_norm[mod_name].setdefault(layer_idx, [])

            if path == "":
                block_inp[mod_name][layer_idx].append(input[0].detach().cpu())
                block_out[mod_name][layer_idx].append(output[0].detach().cpu())
            elif path in ("post_attention_layernorm", "norm2"):
                post_norm[mod_name][layer_idx].append(input[0].detach().cpu())
            else:
                raise ValueError(f"path {path} not supported.")
        return hook_fn

    hooks = []

    config = model.config.to_dict()

    for mod_name, spec in indexing.items():
        cfg_name   = spec["config_attr"]
        num_attr   = spec["num_blocks_attr"]
        paths      = spec["path_list"]
        template   = spec["path_template"]

        num_layers = config[cfg_name][num_attr]

        for i in range(num_layers):
            for path in paths:
                full_path = template.format(block_index=i, path=path)
                module = get_module_by_name(model, full_path)
                hooks.append(module.register_forward_hook(make_hook(mod_name, i, path)))

    return hooks, block_inp, block_out, post_norm


def run_vlm_inference(model, processor, batch):
    """
    Extracted VLM inference routine from eval_bi().
    Performs text+vision preprocessing and model forward pass.
    Returns the inputs dictionary (for hooks).
    """

    # Step 1: Convert chat template
    messages = batch["messages"]
    texts = [processor.apply_chat_template(m, tokenize=False) for m in messages]

    # Step 2: Vision preprocessing
    from qwen_vl_utils import process_vision_info
    vision_data = [process_vision_info(m) for m in messages]
    imgs, vids = zip(*vision_data)

    # None-handling exactly as before
    imgs = None if all(i is None for i in imgs) else list(imgs)
    vids = None if all(v is None for v in vids) else list(vids)

    # Step 3: Processor → model input
    inputs = processor(
        text=texts,
        images=imgs,
        videos=vids,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Step 4: Forward pass (this triggers all hooks)
    _ = model(**inputs)

    return inputs


def eval_bi(model, loader, indexing, processor):

    model.eval()
    nof_batches = 1

    config = model.config.to_dict()

    layer_counts = {
        mod_name: config[spec["config_attr"]][spec["num_blocks_attr"]]
        for mod_name, spec in indexing.items()
    }

    BI = {
        mod_name: {
            i: {"layer_bi": 0, "attn_bi": 0, "mlp_bi": 0, "count": 0}
            for i in range(num_layers)
        }
        for mod_name, num_layers in layer_counts.items()
    }

    with torch.no_grad():
        for batch_i, batch in enumerate(loader):
            if batch_i >= nof_batches:
                break

            hooks, block_inp, block_out, post_norm = register_hooks(model, indexing)

            # ==== extracted inference ====
            _ = run_vlm_inference(model, processor, batch)

            # ==== BI computation follows exactly as before ====
            for mod_name in indexing:
                for layer_idx in block_inp[mod_name]:
                    inp_list  = block_inp[mod_name][layer_idx]
                    out_list  = block_out[mod_name][layer_idx]
                    norm_list = post_norm[mod_name][layer_idx]

                    for x_i, y_i, z_i in zip(inp_list, out_list, norm_list):
                        L, A, M = compute_bi(x_i, z_i, y_i)

                        BI[mod_name][layer_idx]["layer_bi"] += L
                        BI[mod_name][layer_idx]["attn_bi"]  += A
                        BI[mod_name][layer_idx]["mlp_bi"]   += M
                        BI[mod_name][layer_idx]["count"]    += 1

            for h in hooks:
                h.remove()
    
    # ==== Final average ====
    final = {
        mod_name: {
            layer: {
                "layer_bi_score": rec["layer_bi"] / rec["count"] if rec["count"] else 0,
                "attn_bi_score":  rec["attn_bi"]  / rec["count"] if rec["count"] else 0,
                "mlp_bi_score":   rec["mlp_bi"]   / rec["count"] if rec["count"] else 0,
            }
            for layer, rec in mod_recs.items()
        }
        for mod_name, mod_recs in BI.items()
    }

    return final