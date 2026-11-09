from DomURLsBERT import DomURLsBERT


class CySecBERT(DomURLsBERT):
    """CySecBERT text-only sequence classifier."""

    def __init__(
        self,
        output_size: int = 2,
        drop_prob: float = 0.2,
        pretrained_path: str = "markusbayer/CySecBERT",
        **kwargs,
    ):
        super().__init__(
            output_size=output_size,
            drop_prob=drop_prob,
            pretrained_path=pretrained_path,
            **kwargs,
        )


PLMEncoder = CySecBERT
