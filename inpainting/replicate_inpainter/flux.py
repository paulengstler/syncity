from .base import BaseReplicateInpainter


class ReplicateFluxInpainter(BaseReplicateInpainter):
    """
    Replicate playground: https://replicate.com/black-forest-labs/flux-fill-dev
    Speed and Cost: 0.04$ / image, 9.6s / image
    """

    REPLICATE_ID = "fishwowater/flux-dev-controlnet-inpainting-beta:27d3ff35f58b4409775de5a0b36e99b4c6d2d7fc7fe772b35170951db678ec63"

    def _build_extra_inputs(self):
        return {
            # default guidance scale
            "guidance_scale": 3.5,
            "true_guidance_scale": 3.5,
            "controlnet_conditioning_scale": 0.9,
            # default values for inference steps
            "num_inference_steps": 24,
            "output_quality": 100,
        }
