from .base import BaseReplicateInpainter



class ReplicateSDXLInpainter(BaseReplicateInpainter):
    """
    Replicate playground: https://replicate.com/lucataco/sdxl-inpainting
    Speed and Cost: 0.0023$ / image, 1.9s / image
    """

    REPLICATE_ID = "lucataco/sdxl-inpainting:a5b13068cc81a89a4fbeefeccc774869fcb34df4dbc92c1555e0f2771d49dde7"

    def _build_extra_inputs(self):
        return {
            # default values on the playground
            "guidance_scale": 8.0,
            "steps": 20,
            "strength": 0.7,
        }