import os, json, shutil
from huggingface_hub import HfApi, create_repo
from transformers import PreTrainedModel

def _fqcn(cls): # fully-qualified class name
    return f"{cls.__module__}.{cls.__name__}"

def _infer_auto_map(model: PreTrainedModel):
    # always map config and generic AutoModel
    auto_map = {
        "AutoConfig": _fqcn(model.config.__class__),
        "AutoModel": _fqcn(model.__class__),
    }
    # Heuristics for common specialized autos
    name = model.__class__.__name__
    if "ForConditionalGeneration" in name:
        auto_map["AutoModelForConditionalGeneration"] = _fqcn(model.__class__)
    ### add more heuristics here if needed
    ### ...

    return auto_map

def export_to_hf(
        model: PreTrainedModel,
        repo_id: str,
        base_model: str = None,
        readme: str = None,
        out_dir = None,
        embed_code = False,
        token = None,
        private = False,
        exist_ok = False,
):
    # Setup output directory
    out_dir = os.path.join(out_dir or "", f"{repo_id.split('/')[-1]}")
    os.makedirs(out_dir, exist_ok=True)

    # save the model on disk
    model.save_pretrained(out_dir)

    # [optional] save tokenizer/processor from a base model (helps users load)
    if base_model:
        try:
            from transformers import AutoTokenizer, AutoProcessor
            AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
        except Exception:
            pass
        try:
            from transformers import AutoProcessor
            AutoProcessor.from_pretrained(base_model).save_pretrained(out_dir)
        except Exception:
            pass

    # [optional] save README
    if readme:
        with open(os.path.join(out_dir, "README.md"), "w") as f:
            f.write(readme)

    # trust_remote_code and optinal embedded package for custom classes
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f: cfg = json.load(f)
        cfg["trust_remote_code"] = True
        cfg["auto_map"] = _infer_auto_map(model)
        with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)

    if embed_code:
        try:
            import transformersurgeon as ts
            shutil.copytree(ts.__path__[0], os.path.join(out_dir, "transformersurgeon"))
        except Exception as e:
            print(f"Warning: could not embed transformersurgeon package: {e}")

    # [optional] upload to Hugging Face Hub
    if token:
        api = HfApi(token=token)
        try:
            create_repo(repo_id=repo_id, token=token, private=private, exist_ok=exist_ok)
        except Exception as e:
            print(f"Warning: could not create repo (it may already exist): {e}")
        api.upload_folder(
            folder_path=out_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            ignore_patterns=["__pycache__",],
            commit_message="Upload model",
        )

    return out_dir
            