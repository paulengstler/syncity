import os, os.path as osp 
import replicate
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

TMP_DIR = "./tmp_flux"
TMP_PATH = osp.join(TMP_DIR, "result.png")

class BaseReplicateInpainter(ABC):
    REPLICATE_ID = ""
    def __init__(self) -> None:
        pass

    @abstractmethod
    def _build_extra_inputs(self):
        pass

    def run(self, image: Image, mask: Image, seed: int, prompt: str):
        inputs = {
            "image": image,
            "mask": mask,
            "seed": seed,
            "prompt": prompt,
        }
        inputs.update(self._build_extra_inputs())
        print(inputs.keys())
        return replicate.run(
            self.REPLICATE_ID,
            input=inputs
        )
    
    def __call__(self, image: Image, mask: Image, seed: int, prompt: str):
        os.makedirs(TMP_DIR, exist_ok=True)
        image = image.convert("RGB")
        mask_rgb = Image.new('RGB', mask.size)
        mask_rgb.paste(mask)
        image_tmp_path = osp.join(TMP_DIR, "image.png")
        mask_tmp_path = osp.join(TMP_DIR, "mask.png")
        image.save(image_tmp_path)
        mask_rgb.save(mask_tmp_path)

        print(np.array(mask_rgb).shape, np.array(image).shape)
        # exit(0)

        output = self.run(open(image_tmp_path, "rb"), open(mask_tmp_path, "rb"), seed, prompt)
        # output = self.run(image, mask, seed, prompt)
        assert len(output) == 1

        with open(TMP_PATH, "wb") as file:
            file.write(output[0].read())
        return Image.open(TMP_PATH)