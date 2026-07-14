from .decoders import LinearDecoder, RidgeDecoder


def build_decoder(cfg):
    decoder_type = cfg["train"]["type"].lower()

    if decoder_type == "linear":
        return LinearDecoder()

    if decoder_type == "ridge":
        alpha = cfg["train"].get("alpha", 1.0)
        return RidgeDecoder(alpha=alpha)

    raise ValueError(f"Unknown decoder {decoder_type}")