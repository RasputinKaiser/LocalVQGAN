SUPPORTED_CHECKPOINTS = {"imagenet_1024", "imagenet_16384", "wikiart_1024",
                         "wikiart_16384", "coco", "sflckr"}
SUPPORTED_CLIP = {"ViT-B-32", "ViT-B-16"}


def supports(checkpoint: str, clip_model: str) -> bool:
    return checkpoint in SUPPORTED_CHECKPOINTS and clip_model in SUPPORTED_CLIP
