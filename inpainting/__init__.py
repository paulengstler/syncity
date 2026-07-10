from dotenv import load_dotenv
from typing import Literal
from PIL import Image
from .flux_inpainter_server.inpaint import Inpainter as FluxInpainter 
from .replicate_inpainter import ReplicateFluxInpainter, ReplicateSDXLInpainter

# load api keys of replicate
load_dotenv()

class Inpainter:
    def __init__(self, inpainter_type: Literal["flux_local", "flux_replicate", "sdxl_replicate"] = "flux_local", gradio_url: str = ""):
        if inpainter_type == "flux_local":
            self.inpainter = FluxInpainter(gradio_url)
        elif inpainter_type == "flux_replicate":
            self.inpainter = ReplicateFluxInpainter()
        elif inpainter_type == "sdxl_replicate":
            self.inpainter = ReplicateSDXLInpainter()
        else:
            raise ValueError(f"Invalid inpainter_type: {inpainter_type}")

    def __call__(self,  image:Image.Image, mask:Image.Image, seed:int, prompt:str):
        return self.inpainter(image, mask, seed, prompt)
