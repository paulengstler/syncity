import gradio as gr
import torch
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline

check_min_version("0.30.2")

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
)


# Build pipeline
controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta", torch_dtype=torch.bfloat16)
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16
)
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)


MARKDOWN = """
# FLUX.1-dev-Inpainting-Model-Beta-GPU 🔥
Model by alimama-creative
"""

def process(input_image_editor,
            prompt,
            negative_prompt,
            controlnet_conditioning_scale,
            guidance_scale,
            seed,
            num_inference_steps,
            true_guidance_scale            
            ):
    # move model to GPU
    controlnet.to("cuda")
    transformer.to("cuda")
    pipe.to("cuda")

    image = input_image_editor['background']
    mask = input_image_editor['layers'][0]
    size = (768, 768)
    image_or = image.copy()
    
    image = image.convert("RGB").resize(size)
    mask = mask.convert("RGB").resize(size)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        true_guidance_scale=true_guidance_scale
    ).images[0]

    # you may remove this if you have a large gpu or are running it
    # on a dedicated server
    # move model off GPU
    controlnet.to("cpu")
    transformer.to("cpu")
    pipe.to("cpu")

    torch.cuda.empty_cache()

    return result.resize((image_or.size[:2]))


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image_editor_component = gr.ImageEditor(
                label='Image',
                type='pil',
                sources=["upload", "webcam"],
                image_mode='RGB',
                layers=False,
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))

            prompt = gr.Textbox(lines=2, placeholder="Enter prompt here...")
            negative_prompt = gr.Textbox(lines=2, placeholder="Enter negative_prompt here...")
            controlnet_conditioning_scale = gr.Slider(minimum=0, step=0.01, maximum=1, value=0.9, label="controlnet_conditioning_scale")
            guidance_scale = gr.Slider(minimum=1, step=0.5, maximum=10, value=3.5, label="Image to generate")
            seed  = gr.Slider(minimum=0, step=1, maximum=10000000, value=124, label="Seed Value")
            num_inference_steps = gr.Slider(minimum=1, step=1, maximum=30, value=24, label="num_inference_steps")
            true_guidance_scale = gr.Slider(minimum=1, step=1, maximum=10, value=3.5, label="true_guidance_scale")
            

            
            submit_button_component = gr.Button(
                    value='Submit', variant='primary', scale=0)
            
        with gr.Column():
            output_image_component = gr.Image(
                type='pil', image_mode='RGB', label='Generated image', format="png")

    submit_button_component.click(
        fn=process,
        inputs=[
            input_image_editor_component,
            prompt,
            negative_prompt,
            controlnet_conditioning_scale,
            guidance_scale,
            seed,
            num_inference_steps,
            true_guidance_scale  

        ],
        outputs=[
            output_image_component,
        ]
    )

demo.launch(debug=True, show_error=True)