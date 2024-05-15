task_stablepy = {
    'txt2img': 'txt2img',
    'img2img': 'img2img',
    'inpaint': 'inpaint',
    'sd_openpose ControlNet': 'openpose',
    'sd_canny ControlNet': 'canny',
    'sd_mlsd ControlNet': 'mlsd',
    'sd_scribble ControlNet': 'scribble',
    'sd_softedge ControlNet': 'softedge',
    'sd_segmentation ControlNet': 'segmentation',
    'sd_depth ControlNet': 'depth',
    'sd_normalbae ControlNet': 'normalbae',
    'sd_lineart ControlNet': 'lineart',
    'sd_lineart_anime ControlNet': 'lineart_anime',
    'sd_shuffle ControlNet': 'shuffle',
    'sd_ip2p ControlNet': 'ip2p',
    'sdxl_canny T2I Adapter': 'sdxl_canny',
    'sdxl_sketch  T2I Adapter': 'sdxl_sketch',
    'sdxl_lineart  T2I Adapter': 'sdxl_lineart',
    'sdxl_depth-midas  T2I Adapter': 'sdxl_depth-midas',
    'sdxl_openpose  T2I Adapter': 'sdxl_openpose'
}

task_model_list = list(task_stablepy.keys())

#######################
# UTILS
#######################
import spaces
import os
from stablepy import Model_Diffusers
from stablepy.diffusers_vanilla.model import scheduler_names
from stablepy.diffusers_vanilla.style_prompt_config import STYLE_NAMES
import torch
import re

preprocessor_controlnet = {
  "openpose": [
    "Openpose",
    "None",
  ],
  "scribble": [
    "HED",
    "Pidinet",
    "None",
  ],
  "softedge": [
    "Pidinet",
    "HED",
    "HED safe",
    "Pidinet safe",
    "None",
  ],
  "segmentation": [
    "UPerNet",
    "None",
  ],
  "depth": [
    "DPT",
    "Midas",
    "None",
  ],
  "normalbae": [
    "NormalBae",
    "None",
  ],
  "lineart": [
    "Lineart",
    "Lineart coarse",
    "LineartAnime",
    "None",
    "None (anime)",
  ],
  "shuffle": [
    "ContentShuffle",
    "None",
  ],
  "canny": [
    "Canny"
  ],
  "mlsd": [
    "MLSD"
  ],
  "ip2p": [
    "ip2p"
  ]
}


def download_things(directory, url, hf_token="", civitai_api_key=""):
    url = url.strip()

    if "drive.google.com" in url:
        original_dir = os.getcwd()
        os.chdir(directory)
        os.system(f"gdown --fuzzy {url}")
        os.chdir(original_dir)
    elif "huggingface.co" in url:
        url = url.replace("?download=true", "")
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        user_header = f'"Authorization: Bearer {hf_token}"'
        if hf_token:
            os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 {url} -d {directory}  -o {url.split('/')[-1]}")
        else:
            os.system (f"aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 {url} -d {directory}  -o {url.split('/')[-1]}")
    elif "civitai.com" in url:
        if "?" in url:
            url = url.split("?")[0]
        if civitai_api_key:
            url = url + f"?token={civitai_api_key}"
            os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {directory} {url}")
        else:
            print("\033[91mYou need an API key to download Civitai models.\033[0m")
    else:
        os.system(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {directory} {url}")


def get_model_list(directory_path):
    model_list = []
    valid_extensions = {'.ckpt' , '.pt', '.pth', '.safetensors', '.bin'}

    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1] in valid_extensions:
            name_without_extension = os.path.splitext(filename)[0]
            file_path = os.path.join(directory_path, filename)
            # model_list.append((name_without_extension, file_path))
            model_list.append(file_path)
            print('\033[34mFILE: ' + file_path + '\033[0m')
    return model_list


def process_string(input_string):
    parts = input_string.split('/')

    if len(parts) == 2:
        first_element = parts[1]
        complete_string = input_string
        result = (first_element, complete_string)
        return result
    else:
        return None


directory_models = 'models'
os.makedirs(directory_models, exist_ok=True)
directory_loras = 'loras'
os.makedirs(directory_loras, exist_ok=True)
directory_vaes = 'vaes'
os.makedirs(directory_vaes, exist_ok=True)

# - **Download SD 1.5 Models**
download_model = "https://huggingface.co/frankjoshua/toonyou_beta6/resolve/main/toonyou_beta6.safetensors"
# - **Download VAEs**
download_vae = "https://huggingface.co/fp16-guy/anything_kl-f8-anime2_vae-ft-mse-840000-ema-pruned_blessed_clearvae_fp16_cleaned/resolve/main/anything_fp16.safetensors"
# - **Download LoRAs**
download_lora = "https://civitai.com/api/download/models/97655, https://civitai.com/api/download/models/124358"
load_diffusers_format_model = ['stabilityai/stable-diffusion-xl-base-1.0', 'runwayml/stable-diffusion-v1-5']
CIVITAI_API_KEY = ""
hf_token = ""

# Download stuffs
for url in [url.strip() for url in download_model.split(',')]:
    if not os.path.exists(f"./models/{url.split('/')[-1]}"):
        download_things(directory_models, url, hf_token, CIVITAI_API_KEY)
for url in [url.strip() for url in download_vae.split(',')]:
    if not os.path.exists(f"./vaes/{url.split('/')[-1]}"):
        download_things(directory_vaes, url, hf_token, CIVITAI_API_KEY)
for url in [url.strip() for url in download_lora.split(',')]:
    if not os.path.exists(f"./loras/{url.split('/')[-1]}"):
        download_things(directory_loras, url, hf_token, CIVITAI_API_KEY)

# Download Embeddings
directory_embeds = 'embedings'
os.makedirs(directory_embeds, exist_ok=True)
download_embeds = [
    'https://huggingface.co/datasets/Nerfgun3/bad_prompt/resolve/main/bad_prompt.pt',
    'https://huggingface.co/datasets/Nerfgun3/bad_prompt/blob/main/bad_prompt_version2.pt',
    'https://huggingface.co/embed/EasyNegative/resolve/main/EasyNegative.safetensors',
    'https://huggingface.co/embed/negative/resolve/main/EasyNegativeV2.safetensors',
    'https://huggingface.co/embed/negative/resolve/main/bad-hands-5.pt',
    'https://huggingface.co/embed/negative/resolve/main/bad-artist.pt',
    'https://huggingface.co/embed/negative/resolve/main/ng_deepnegative_v1_75t.pt',
    'https://huggingface.co/embed/negative/resolve/main/bad-artist-anime.pt',
    'https://huggingface.co/embed/negative/resolve/main/bad-image-v2-39000.pt',
    'https://huggingface.co/embed/negative/resolve/main/verybadimagenegative_v1.3.pt',
    ]

for url_embed in download_embeds:
    if not os.path.exists(f"./embedings/{url_embed.split('/')[-1]}"):
        download_things(directory_embeds, url_embed, hf_token, CIVITAI_API_KEY)

# Build list models
embed_list = get_model_list(directory_embeds)
model_list = get_model_list(directory_models)
model_list = load_diffusers_format_model + model_list
lora_model_list = get_model_list(directory_loras)
lora_model_list.insert(0, "None")
vae_model_list = get_model_list(directory_vaes)
vae_model_list.insert(0, "None")

print('\033[33m🏁 Download and listing of valid models completed.\033[0m')

upscaler_dict_gui = {
    None : None,
    "Lanczos" : "Lanczos",
    "Nearest" : "Nearest",
    "RealESRGAN_x4plus" : "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus" : "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3" : "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "4x-UltraSharp" : "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "4x_foolhardy_Remacri" : "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "Remacri4xExtraSmoother" : "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    "AnimeSharp4x" : "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "lollypop" : "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    "RealisticRescaler4x" : "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    "NickelbackFS4x" : "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth"
}


def extract_parameters(input_string):
    parameters = {}
    input_string = input_string.replace("\n", "")

    if not "Negative prompt:" in input_string:
        print("Negative prompt not detected")
        parameters["prompt"] = input_string
        return parameters

    parm = input_string.split("Negative prompt:")
    parameters["prompt"] = parm[0]
    if not "Steps:" in parm[1]:
        print("Steps not detected")
        parameters["neg_prompt"] = parm[1]
        return parameters
    parm = parm[1].split("Steps:")
    parameters["neg_prompt"] = parm[0]
    input_string = "Steps:" + parm[1]

    # Extracting Steps
    steps_match = re.search(r'Steps: (\d+)', input_string)
    if steps_match:
        parameters['Steps'] = int(steps_match.group(1))

    # Extracting Size
    size_match = re.search(r'Size: (\d+x\d+)', input_string)
    if size_match:
        parameters['Size'] = size_match.group(1)
        width, height = map(int, parameters['Size'].split('x'))
        parameters['width'] = width
        parameters['height'] = height

    # Extracting other parameters
    other_parameters = re.findall(r'(\w+): (.*?)(?=, \w+|$)', input_string)
    for param in other_parameters:
        parameters[param[0]] = param[1].strip('"')

    return parameters


#######################
# GUI
#######################
import spaces
import gradio as gr
from PIL import Image
import IPython.display
import time, json
from IPython.utils import capture
import logging
logging.getLogger("diffusers").setLevel(logging.ERROR)
import diffusers
diffusers.utils.logging.set_verbosity(40)
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=UserWarning, module="diffusers")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="transformers")
from stablepy import logger
logger.setLevel(logging.DEBUG)


class GuiSD:
    def __init__(self):
        self.model = None

    @spaces.GPU
    def infer(self, model, pipe_params):
        images, image_list = model(**pipe_params)
        return images

    # @spaces.GPU
    def generate_pipeline(
        self,
        prompt,
        neg_prompt,
        num_images,
        steps,
        cfg,
        clip_skip,
        seed,
        lora1,
        lora_scale1,
        lora2,
        lora_scale2,
        lora3,
        lora_scale3,
        lora4,
        lora_scale4,
        lora5,
        lora_scale5,
        sampler,
        img_height,
        img_width,
        model_name,
        vae_model,
        task,
        image_control,
        preprocessor_name,
        preprocess_resolution,
        image_resolution,
        style_prompt,  # list []
        style_json_file,
        image_mask,
        strength,
        low_threshold,
        high_threshold,
        value_threshold,
        distance_threshold,
        controlnet_output_scaling_in_unet,
        controlnet_start_threshold,
        controlnet_stop_threshold,
        textual_inversion,
        syntax_weights,
        loop_generation,
        leave_progress_bar,
        disable_progress_bar,
        image_previews,
        display_images,
        save_generated_images,
        image_storage_location,
        retain_compel_previous_load,
        retain_detailfix_model_previous_load,
        retain_hires_model_previous_load,
        t2i_adapter_preprocessor,
        t2i_adapter_conditioning_scale,
        t2i_adapter_conditioning_factor,
        upscaler_model_path,
        upscaler_increases_size,
        esrgan_tile,
        esrgan_tile_overlap,
        hires_steps,
        hires_denoising_strength,
        hires_sampler,
        hires_prompt,
        hires_negative_prompt,
        hires_before_adetailer,
        hires_after_adetailer,
        xformers_memory_efficient_attention,
        freeu,
        generator_in_cpu,
        adetailer_inpaint_only,
        adetailer_verbose,
        adetailer_sampler,
        adetailer_active_a,
        prompt_ad_a,
        negative_prompt_ad_a,
        strength_ad_a,
        face_detector_ad_a,
        person_detector_ad_a,
        hand_detector_ad_a,
        mask_dilation_a,
        mask_blur_a,
        mask_padding_a,
        adetailer_active_b,
        prompt_ad_b,
        negative_prompt_ad_b,
        strength_ad_b,
        face_detector_ad_b,
        person_detector_ad_b,
        hand_detector_ad_b,
        mask_dilation_b,
        mask_blur_b,
        mask_padding_b,
    ):
        
        task = task_stablepy[task]

        # First load
        model_precision = torch.float16
        if not self.model:
            from stablepy import Model_Diffusers

            print("Loading model...")
            self.model = Model_Diffusers(
                base_model_id=model_name,
                task_name=task,
                vae_model=vae_model if vae_model != "None" else None,
                type_model_precision=model_precision
            )

        self.model.load_pipe(
            model_name,
            task_name=task,
            vae_model=vae_model if vae_model != "None" else None,
            type_model_precision=model_precision
        )

        if task != "txt2img" and not image_control:
            raise ValueError("No control image found: To use this function, you have to upload an image in 'Image ControlNet/Inpaint/Img2img'")

        if task == "inpaint" and not image_mask:
            raise ValueError("No mask image found: Specify one in 'Image Mask'")

        if upscaler_model_path in [None, "Lanczos", "Nearest"]:
            upscaler_model = upscaler_model_path
        else:
            directory_upscalers = 'upscalers'
            os.makedirs(directory_upscalers, exist_ok=True)

            url_upscaler = upscaler_dict_gui[upscaler_model_path]

            if not os.path.exists(f"./upscalers/{url_upscaler.split('/')[-1]}"):
                download_things(directory_upscalers, url_upscaler, hf_token)

            upscaler_model = f"./upscalers/{url_upscaler.split('/')[-1]}"

        if textual_inversion and self.model.class_name == "StableDiffusionXLPipeline":
            print("No Textual inversion for SDXL")

        logging.getLogger("ultralytics").setLevel(logging.INFO if adetailer_verbose else logging.ERROR)

        adetailer_params_A = {
            "face_detector_ad" : face_detector_ad_a,
            "person_detector_ad" : person_detector_ad_a,
            "hand_detector_ad" : hand_detector_ad_a,
            "prompt": prompt_ad_a,
            "negative_prompt" : negative_prompt_ad_a,
            "strength" : strength_ad_a,
            # "image_list_task" : None,
            "mask_dilation" : mask_dilation_a,
            "mask_blur" : mask_blur_a,
            "mask_padding" : mask_padding_a,
            "inpaint_only" : adetailer_inpaint_only,
            "sampler" : adetailer_sampler,
        }

        adetailer_params_B = {
            "face_detector_ad" : face_detector_ad_b,
            "person_detector_ad" : person_detector_ad_b,
            "hand_detector_ad" : hand_detector_ad_b,
            "prompt": prompt_ad_b,
            "negative_prompt" : negative_prompt_ad_b,
            "strength" : strength_ad_b,
            # "image_list_task" : None,
            "mask_dilation" : mask_dilation_b,
            "mask_blur" : mask_blur_b,
            "mask_padding" : mask_padding_b,
        }
        pipe_params = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "img_height": img_height,
            "img_width": img_width,
            "num_images": num_images,
            "num_steps": steps,
            "guidance_scale": cfg,
            "clip_skip": clip_skip,
            "seed": seed,
            "image": image_control,
            "preprocessor_name": preprocessor_name,
            "preprocess_resolution": preprocess_resolution,
            "image_resolution": image_resolution,
            "style_prompt": style_prompt if style_prompt else "",
            "style_json_file": "",
            "image_mask": image_mask,  # only for Inpaint
            "strength": strength,  # only for Inpaint or ...
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "value_threshold": value_threshold,
            "distance_threshold": distance_threshold,
            "lora_A": lora1 if lora1 != "None" else None,
            "lora_scale_A": lora_scale1,
            "lora_B": lora2 if lora2 != "None" else None,
            "lora_scale_B": lora_scale2,
            "lora_C": lora3 if lora3 != "None" else None,
            "lora_scale_C": lora_scale3,
            "lora_D": lora4 if lora4 != "None" else None,
            "lora_scale_D": lora_scale4,
            "lora_E": lora5 if lora5 != "None" else None,
            "lora_scale_E": lora_scale5,
            "textual_inversion": embed_list if textual_inversion and self.model.class_name != "StableDiffusionXLPipeline" else [],
            "syntax_weights": syntax_weights,  # "Classic"
            "sampler": sampler,
            "xformers_memory_efficient_attention": xformers_memory_efficient_attention,
            "gui_active": True,
            "loop_generation": loop_generation,
            "controlnet_conditioning_scale": float(controlnet_output_scaling_in_unet),
            "control_guidance_start": float(controlnet_start_threshold),
            "control_guidance_end": float(controlnet_stop_threshold),
            "generator_in_cpu": generator_in_cpu,
            "FreeU": freeu,
            "adetailer_A": adetailer_active_a,
            "adetailer_A_params": adetailer_params_A,
            "adetailer_B": adetailer_active_b,
            "adetailer_B_params": adetailer_params_B,
            "leave_progress_bar": leave_progress_bar,
            "disable_progress_bar": disable_progress_bar,
            "image_previews": image_previews,
            "display_images": display_images,
            "save_generated_images": save_generated_images,
            "image_storage_location": image_storage_location,
            "retain_compel_previous_load": retain_compel_previous_load,
            "retain_detailfix_model_previous_load": retain_detailfix_model_previous_load,
            "retain_hires_model_previous_load": retain_hires_model_previous_load,
            "t2i_adapter_preprocessor": t2i_adapter_preprocessor,
            "t2i_adapter_conditioning_scale": float(t2i_adapter_conditioning_scale),
            "t2i_adapter_conditioning_factor": float(t2i_adapter_conditioning_factor),
            "upscaler_model_path": upscaler_model,
            "upscaler_increases_size": upscaler_increases_size,
            "esrgan_tile": esrgan_tile,
            "esrgan_tile_overlap": esrgan_tile_overlap,
            "hires_steps": hires_steps,
            "hires_denoising_strength": hires_denoising_strength,
            "hires_prompt": hires_prompt,
            "hires_negative_prompt": hires_negative_prompt,
            "hires_sampler": hires_sampler,
            "hires_before_adetailer": hires_before_adetailer,
            "hires_after_adetailer": hires_after_adetailer
        }

        # print(pipe_params)

        return self.infer(self.model, pipe_params)


sd_gen = GuiSD()

title_tab_one = "<h2 style='color: #2C5F2D;'>SD Interactive</h2>"
title_tab_adetailer = "<h2 style='color: #97BC62;'>Adetailer</h2>"
title_tab_hires = "<h2 style='color: #97BC62;'>High-resolution</h2>"
title_tab_settings = "<h2 style='color: #97BC62;'>Settings</h2>"

CSS ="""
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#gallery { flex-grow: 1; }
"""

with gr.Blocks(theme="NoCrypt/miku", css=CSS) as app:
    gr.Markdown("# 🧩 DiffuseCraft")
    gr.Markdown(
        f"""
            ### This demo uses [diffusers](https://github.com/huggingface/diffusers) to perform different tasks in image generation.
            """
    )
    with gr.Tab("Generation"):
        with gr.Row():
    
            with gr.Column(scale=2):
                task_gui = gr.Dropdown(label="Task", choices=task_model_list, value=task_model_list[0])
                model_name_gui = gr.Dropdown(label="Model", choices=model_list, value=model_list[0], allow_custom_value=True)
                prompt_gui = gr.Textbox(lines=5, placeholder="Enter prompt")
                neg_prompt_gui = gr.Textbox(lines=3, placeholder="Enter Neg prompt")
                generate_button = gr.Button(value="GENERATE", variant="primary")
    
                result_images = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                    columns=[2],
                    rows=[3],
                    object_fit="contain",
                    # height="auto",
                    interactive=False,
                    preview=True,
                    selected_index=50,
                )
    
            with gr.Column(scale=1):
                steps_gui = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Steps")
                cfg_gui = gr.Slider(minimum=0, maximum=30, step=0.5, value=7.5, label="CFG")
                sampler_gui = gr.Dropdown(label="Sampler", choices=scheduler_names, value="Euler a")
                img_height_gui = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Height")
                img_width_gui = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Width")
                clip_skip_gui = gr.Checkbox(value=True, label="Layer 2 Clip Skip")
                free_u_gui = gr.Checkbox(value=True, label="FreeU")
                seed_gui = gr.Number(minimum=-1, maximum=9999999999, value=-1, label="Seed")
                num_images_gui = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Images")
                prompt_s_options = [("Compel (default) format: (word)weight", "Compel"), ("Classic (sd1.5 long prompts) format: (word:weight)", "Classic")]
                prompt_syntax_gui = gr.Dropdown(label="Prompt Syntax", choices=prompt_s_options, value=prompt_s_options[0][1])
                vae_model_gui = gr.Dropdown(label="VAE Model", choices=vae_model_list)
    
                with gr.Accordion("ControlNet / Img2img / Inpaint", open=False, visible=True):
                    image_control = gr.Image(label="Image ControlNet/Inpaint/Img2img", type="filepath")
                    image_mask_gui = gr.Image(label="Image Mask", type="filepath")
                    strength_gui = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.35, label="Strength")
                    image_resolution_gui = gr.Slider(minimum=64, maximum=2048, step=64, value=1024, label="Image Resolution")
                    preprocessor_name_gui = gr.Dropdown(label="Preprocessor Name", choices=preprocessor_controlnet["canny"])
    
                    def change_preprocessor_choices(task):
                        if task in preprocessor_controlnet.keys():
                            choices_task = preprocessor_controlnet[task]
                        else:
                            choices_task = preprocessor_controlnet["canny"]
                        return gr.update(choices=choices_task, value=choices_task[0])
    
                    task_gui.change(
                        change_preprocessor_choices,
                        [task_gui],
                        [preprocessor_name_gui],
                    )
                    preprocess_resolution_gui = gr.Slider(minimum=64, maximum=2048, step=64, value=512, label="Preprocess Resolution")
                    low_threshold_gui = gr.Slider(minimum=1, maximum=255, step=1, value=100, label="Canny low threshold")
                    high_threshold_gui = gr.Slider(minimum=1, maximum=255, step=1, value=200, label="Canny high threshold")
                    value_threshold_gui = gr.Slider(minimum=1, maximum=2.0, step=0.01, value=0.1, label="Hough value threshold (MLSD)")
                    distance_threshold_gui = gr.Slider(minimum=1, maximum=20.0, step=0.01, value=0.1, label="Hough distance threshold (MLSD)")
                    control_net_output_scaling_gui = gr.Slider(minimum=0, maximum=5.0, step=0.1, value=1, label="ControlNet Output Scaling in UNet")
                    control_net_start_threshold_gui = gr.Slider(minimum=0, maximum=1, step=0.01, value=0, label="ControlNet Start Threshold (%)")
                    control_net_stop_threshold_gui = gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="ControlNet Stop Threshold (%)")

                with gr.Accordion("T2I adapter", open=False, visible=True):
                    t2i_adapter_preprocessor_gui = gr.Checkbox(value=True, label="T2i Adapter Preprocessor")
                    adapter_conditioning_scale_gui = gr.Slider(minimum=0, maximum=5., step=0.1, value=1, label="Adapter Conditioning Scale")
                    adapter_conditioning_factor_gui = gr.Slider(minimum=0, maximum=1., step=0.01, value=0.55, label="Adapter Conditioning Factor (%)")

                with gr.Accordion("LoRA", open=False, visible=False):
                    lora1_gui = gr.Dropdown(label="Lora1", choices=lora_model_list)
                    lora_scale_1_gui = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1, label="Lora Scale 1")
                    lora2_gui = gr.Dropdown(label="Lora2", choices=lora_model_list)
                    lora_scale_2_gui = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1, label="Lora Scale 2")
                    lora3_gui = gr.Dropdown(label="Lora3", choices=lora_model_list)
                    lora_scale_3_gui = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1, label="Lora Scale 3")
                    lora4_gui = gr.Dropdown(label="Lora4", choices=lora_model_list)
                    lora_scale_4_gui = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1, label="Lora Scale 4")
                    lora5_gui = gr.Dropdown(label="Lora5", choices=lora_model_list)
                    lora_scale_5_gui = gr.Slider(minimum=-2, maximum=2, step=0.01, value=1, label="Lora Scale 5")
    
                with gr.Accordion("Styles", open=False, visible=True):
                    
                    try:
                        style_names_found = sd_gen.model.STYLE_NAMES
                    except:
                        style_names_found = STYLE_NAMES
                        
                    style_prompt_gui = gr.Dropdown(
                        style_names_found,
                        multiselect=True,
                        value=None,
                        label="Style Prompt",
                        interactive=True,
                    )
                    style_json_gui = gr.File(label="Style JSON File")
                    style_button = gr.Button("Load styles")

                    def load_json_style_file(json):
                        if not sd_gen.model:
                            gr.Info("First load the model")
                            return gr.update(value=None, choices=STYLE_NAMES)
                        
                        sd_gen.model.load_style_file(json)
                        gr.Info(f"{len(sd_gen.model.STYLE_NAMES)} styles loaded")
                        return gr.update(value=None, choices=sd_gen.model.STYLE_NAMES)

                    style_button.click(load_json_style_file, [style_json_gui], [style_prompt_gui])                        
    
                with gr.Accordion("Textual inversion", open=False, visible=False):
                    active_textual_inversion_gui = gr.Checkbox(value=False, label="Active Textual Inversion in prompt")
    
                with gr.Accordion("Hires fix", open=False, visible=False):
    
                    upscaler_keys = list(upscaler_dict_gui.keys())
    
                    upscaler_model_path_gui = gr.Dropdown(label="Upscaler", choices=upscaler_keys, value=upscaler_keys[0])
                    upscaler_increases_size_gui = gr.Slider(minimum=1.1, maximum=6., step=0.1, value=1.5, label="Upscale by")
                    esrgan_tile_gui = gr.Slider(minimum=0, value=100, maximum=500, step=1, label="ESRGAN Tile")
                    esrgan_tile_overlap_gui = gr.Slider(minimum=1, maximum=200, step=1, value=10, label="ESRGAN Tile Overlap")
                    hires_steps_gui = gr.Slider(minimum=0, value=30, maximum=100, step=1, label="Hires Steps")
                    hires_denoising_strength_gui = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.55, label="Hires Denoising Strength")
                    hires_sampler_gui = gr.Dropdown(label="Hires Sampler", choices=["Use same sampler"] + scheduler_names[:-1], value="Use same sampler")
                    hires_prompt_gui = gr.Textbox(label="Hires Prompt", placeholder="Main prompt will be use", lines=3)
                    hires_negative_prompt_gui = gr.Textbox(label="Hires Negative Prompt", placeholder="Main negative prompt will be use", lines=3)
    
                with gr.Accordion("Detailfix", open=False, visible=False):
    
                    # Adetailer Inpaint Only
                    adetailer_inpaint_only_gui = gr.Checkbox(label="Inpaint only", value=True)
    
                    # Adetailer Verbose
                    adetailer_verbose_gui = gr.Checkbox(label="Verbose", value=False)
    
                    # Adetailer Sampler
                    adetailer_sampler_options = ["Use same sampler"] + scheduler_names[:-1]
                    adetailer_sampler_gui = gr.Dropdown(label="Adetailer sampler:", choices=adetailer_sampler_options, value="Use same sampler")
    
                    with gr.Accordion("Detailfix A", open=False, visible=True):
                        # Adetailer A
                        adetailer_active_a_gui = gr.Checkbox(label="Enable Adetailer A", value=False)
                        prompt_ad_a_gui = gr.Textbox(label="Main prompt", placeholder="Main prompt will be use", lines=3)
                        negative_prompt_ad_a_gui = gr.Textbox(label="Negative prompt", placeholder="Main negative prompt will be use", lines=3)
                        strength_ad_a_gui = gr.Number(label="Strength:", value=0.35, step=0.01, minimum=0.01, maximum=1.0)
                        face_detector_ad_a_gui = gr.Checkbox(label="Face detector", value=True)
                        person_detector_ad_a_gui = gr.Checkbox(label="Person detector", value=True)
                        hand_detector_ad_a_gui = gr.Checkbox(label="Hand detector", value=False)
                        mask_dilation_a_gui = gr.Number(label="Mask dilation:", value=4, minimum=1)
                        mask_blur_a_gui = gr.Number(label="Mask blur:", value=4, minimum=1)
                        mask_padding_a_gui = gr.Number(label="Mask padding:", value=32, minimum=1)
    
                    with gr.Accordion("Detailfix B", open=False, visible=True):
                        # Adetailer B
                        adetailer_active_b_gui = gr.Checkbox(label="Enable Adetailer B", value=False)
                        prompt_ad_b_gui = gr.Textbox(label="Main prompt", placeholder="Main prompt will be use", lines=3)
                        negative_prompt_ad_b_gui = gr.Textbox(label="Negative prompt", placeholder="Main negative prompt will be use", lines=3)
                        strength_ad_b_gui = gr.Number(label="Strength:", value=0.35, step=0.01, minimum=0.01, maximum=1.0)
                        face_detector_ad_b_gui = gr.Checkbox(label="Face detector", value=True)
                        person_detector_ad_b_gui = gr.Checkbox(label="Person detector", value=True)
                        hand_detector_ad_b_gui = gr.Checkbox(label="Hand detector", value=False)
                        mask_dilation_b_gui = gr.Number(label="Mask dilation:", value=4, minimum=1)
                        mask_blur_b_gui = gr.Number(label="Mask blur:", value=4, minimum=1)
                        mask_padding_b_gui = gr.Number(label="Mask padding:", value=32, minimum=1)
    
                with gr.Accordion("Other settings", open=False, visible=False):
                    hires_before_adetailer_gui = gr.Checkbox(value=False, label="Hires Before Adetailer")
                    hires_after_adetailer_gui = gr.Checkbox(value=True, label="Hires After Adetailer")
                    loop_generation_gui = gr.Slider(minimum=1, value=1, label="Loop Generation")
                    leave_progress_bar_gui = gr.Checkbox(value=True, label="Leave Progress Bar")
                    disable_progress_bar_gui = gr.Checkbox(value=False, label="Disable Progress Bar")
                    image_previews_gui = gr.Checkbox(value=False, label="Image Previews")
                    display_images_gui = gr.Checkbox(value=False, label="Display Images")
                    save_generated_images_gui = gr.Checkbox(value=False, label="Save Generated Images")
                    image_storage_location_gui = gr.Textbox(value="./images", label="Image Storage Location")
                    retain_compel_previous_load_gui = gr.Checkbox(value=False, label="Retain Compel Previous Load")
                    retain_detailfix_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Detailfix Model Previous Load")
                    retain_hires_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Hires Model Previous Load")
                    xformers_memory_efficient_attention_gui = gr.Checkbox(value=False, label="Xformers Memory Efficient Attention")
                    generator_in_cpu_gui = gr.Checkbox(value=False, label="Generator in CPU")

    with gr.Tab("Inpaint mask maker", render=True):
        
        def create_mask_now(img, invert):            
            import numpy as np
            import time
            
            time.sleep(0.5)

            transparent_image = img["layers"][0]
            
            # Extract the alpha channel
            alpha_channel = np.array(transparent_image)[:, :, 3]
            
            # Create a binary mask by thresholding the alpha channel
            binary_mask = alpha_channel > 1

            if invert:
                print("Invert")
                # Invert the binary mask so that the drawn shape is white and the rest is black
                binary_mask = np.invert(binary_mask)

            # Convert the binary mask to a 3-channel RGB mask
            rgb_mask = np.stack((binary_mask,) * 3, axis=-1)

            # Convert the mask to uint8
            rgb_mask = rgb_mask.astype(np.uint8) * 255
            
            return img["background"], rgb_mask
            
        with gr.Row():
            with gr.Column(scale=2):
                # image_base = gr.ImageEditor(label="Base image", show_label=True, brush=gr.Brush(colors=["#000000"]))
                image_base = gr.ImageEditor(
                    sources=["upload", "clipboard"],
                    # crop_size="1:1",
                    # enable crop (or disable it)
                    # transforms=["crop"],
                    brush=gr.Brush(
                      default_size="16", # or leave it as 'auto'
                      color_mode="fixed", # 'fixed' hides the user swatches and colorpicker, 'defaults' shows it
                      #default_color="black", # html names are supported
                      colors=[
                        "rgba(0, 0, 0, 1)", # rgb(a)
                        "rgba(0, 0, 0, 0.1)",
                        "rgba(255, 255, 255, 0.1)",
                        # "hsl(360, 120, 120)" # in fact any valid colorstring
                      ]
                    ),
                    eraser=gr.Eraser(default_size="16")
                )
                invert_mask = gr.Checkbox(value=False, label="Invert mask")
                btn = gr.Button("Create mask")
            with gr.Column(scale=1):
                img_source = gr.Image(interactive=False)
                img_result = gr.Image(label="Mask image", show_label=True, interactive=False)
                btn_send = gr.Button("Send to the first tab")

            btn.click(create_mask_now, [image_base, invert_mask], [img_source, img_result])

            def send_img(img_source, img_result):
                return img_source, img_result
            btn_send.click(send_img, [img_source, img_result], [image_control, image_mask_gui])
            
    generate_button.click(
        fn=sd_gen.generate_pipeline,
        inputs=[
            prompt_gui,
            neg_prompt_gui,
            num_images_gui,
            steps_gui,
            cfg_gui,
            clip_skip_gui,
            seed_gui,
            lora1_gui,
            lora_scale_1_gui,
            lora2_gui,
            lora_scale_2_gui,
            lora3_gui,
            lora_scale_3_gui,
            lora4_gui,
            lora_scale_4_gui,
            lora5_gui,
            lora_scale_5_gui,
            sampler_gui,
            img_height_gui,
            img_width_gui,
            model_name_gui,
            vae_model_gui,
            task_gui,
            image_control,
            preprocessor_name_gui,
            preprocess_resolution_gui,
            image_resolution_gui,
            style_prompt_gui,
            style_json_gui,
            image_mask_gui,
            strength_gui,
            low_threshold_gui,
            high_threshold_gui,
            value_threshold_gui,
            distance_threshold_gui,
            control_net_output_scaling_gui,
            control_net_start_threshold_gui,
            control_net_stop_threshold_gui,
            active_textual_inversion_gui,
            prompt_syntax_gui,
            loop_generation_gui,
            leave_progress_bar_gui,
            disable_progress_bar_gui,
            image_previews_gui,
            display_images_gui,
            save_generated_images_gui,
            image_storage_location_gui,
            retain_compel_previous_load_gui,
            retain_detailfix_model_previous_load_gui,
            retain_hires_model_previous_load_gui,
            t2i_adapter_preprocessor_gui,
            adapter_conditioning_scale_gui,
            adapter_conditioning_factor_gui,
            upscaler_model_path_gui,
            upscaler_increases_size_gui,
            esrgan_tile_gui,
            esrgan_tile_overlap_gui,
            hires_steps_gui,
            hires_denoising_strength_gui,
            hires_sampler_gui,
            hires_prompt_gui,
            hires_negative_prompt_gui,
            hires_before_adetailer_gui,
            hires_after_adetailer_gui,
            xformers_memory_efficient_attention_gui,
            free_u_gui,
            generator_in_cpu_gui,
            adetailer_inpaint_only_gui,
            adetailer_verbose_gui,
            adetailer_sampler_gui,
            adetailer_active_a_gui,
            prompt_ad_a_gui,
            negative_prompt_ad_a_gui,
            strength_ad_a_gui,
            face_detector_ad_a_gui,
            person_detector_ad_a_gui,
            hand_detector_ad_a_gui,
            mask_dilation_a_gui,
            mask_blur_a_gui,
            mask_padding_a_gui,
            adetailer_active_b_gui,
            prompt_ad_b_gui,
            negative_prompt_ad_b_gui,
            strength_ad_b_gui,
            face_detector_ad_b_gui,
            person_detector_ad_b_gui,
            hand_detector_ad_b_gui,
            mask_dilation_b_gui,
            mask_blur_b_gui,
            mask_padding_b_gui,
        ],
        outputs=[result_images],
        queue=True,
    )



app.queue()  # default_concurrency_limit=40

app.launch(
    # max_threads=40,
    # share=False,
    show_error=True,
    # quiet=False,
    debug=True,
    # allowed_paths=["./assets/"],
)
