from gradio_client import Client, handle_file
import numpy as np
import shutil
import os
from PIL import Image


class Inpainter:
	def __init__(self, server_url:str):
		self.server_url = server_url
		self.client = Client(server_url)
		
	def __call__(self, image:Image.Image, mask:Image.Image, seed:int, prompt:str):
		return query_server(image, mask, seed, prompt, self.client)
	

def query_server(image:Image.Image, mask:Image.Image, seed:int, prompt:str, client:Client):
	os.makedirs('./tmp_flux', exist_ok=True)
	image.save('./tmp_flux/image_flux.png')
	mask.save('./tmp_flux/mask_flux.png')


	result = client.predict(
			input_image_editor={
				"background":handle_file('./tmp_flux/image_flux.png'),
				"layers":[handle_file('./tmp_flux/mask_flux.png')],
				"composite":None},
			prompt=f"{prompt}",
			negative_prompt="",
			controlnet_conditioning_scale=0.9,
			guidance_scale=3.5,
			seed=seed,
			num_inference_steps=24,
			true_guidance_scale=3.5,
			api_name="/process"
	)

	shutil.move(result, f"./tmp_flux/result.png")
	image_out = Image.open(f"./tmp_flux/result.png")
	return image_out