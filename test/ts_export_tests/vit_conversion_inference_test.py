import torch

from transformersurgeon import ViTForImageClassificationCompress
from transformersurgeon.utils import convert_for_export


MODEL_NAME = "google/vit-base-patch16-224"


def run_test(device: torch.device):
    model = ViTForImageClassificationCompress.from_pretrained(MODEL_NAME)
    model.eval().to(device)

    converted_models = convert_for_export(
        model,
        options={"use_sdpa": True},
        verbose=True,
    )
    encoder = converted_models["vit"].to(device)
    encoder.eval()

    pixel_values = torch.randn(1, 3, 224, 224, device=device)

    with torch.no_grad():
        embeddings = model.vit.embeddings(pixel_values)

        hf_hidden = model.vit.encoder(embeddings).last_hidden_state
        hf_hidden = model.vit.layernorm(hf_hidden)

        ts_hidden = encoder(embeddings)

        hf_logits = model.classifier(hf_hidden[:, 0, :])
        ts_logits = model.classifier(ts_hidden[:, 0, :])

    hidden_diff = (hf_hidden - ts_hidden).abs()
    logits_diff = (hf_logits - ts_logits).abs()

    print("ViT hidden max abs diff:", float(hidden_diff.max().item()))
    print("ViT hidden mean abs diff:", float(hidden_diff.mean().item()))
    print("ViT logits max abs diff:", float(logits_diff.max().item()))
    print("ViT logits mean abs diff:", float(logits_diff.mean().item()))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    run_test(device)
