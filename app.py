import spaces
import os
from stablepy import Model_Diffusers
from stablepy.diffusers_vanilla.style_prompt_config import STYLE_NAMES
from stablepy.diffusers_vanilla.constants import FLUX_CN_UNION_MODES
import torch
import re
from huggingface_hub import HfApi
from stablepy import (
    CONTROLNET_MODEL_IDS,
    VALID_TASKS,
    T2I_PREPROCESSOR_NAME,
    FLASH_LORA,
    SCHEDULER_CONFIG_MAP,
    scheduler_names,
    IP_ADAPTER_MODELS,
    IP_ADAPTERS_SD,
    IP_ADAPTERS_SDXL,
    REPO_IMAGE_ENCODER,
    ALL_PROMPT_WEIGHT_OPTIONS,
    SD15_TASKS,
    SDXL_TASKS,
)
import time
from PIL import ImageFile
# import urllib.parse

ImageFile.LOAD_TRUNCATED_IMAGES = True
print(os.getenv("SPACES_ZERO_GPU"))

# - **Download SD 1.5 Models**
download_model = "https://civitai.com/api/download/models/574369, https://huggingface.co/TechnoByte/MilkyWonderland/resolve/main/milkyWonderland_v40.safetensors"
# - **Download VAEs**
download_vae = "https://huggingface.co/nubby/blessed-sdxl-vae-fp16-fix/resolve/main/sdxl_vae-fp16fix-c-1.1-b-0.5.safetensors?download=true, https://huggingface.co/nubby/blessed-sdxl-vae-fp16-fix/resolve/main/sdxl_vae-fp16fix-blessed.safetensors?download=true, https://huggingface.co/digiplay/VAE/resolve/main/vividReal_v20.safetensors?download=true, https://huggingface.co/fp16-guy/anything_kl-f8-anime2_vae-ft-mse-840000-ema-pruned_blessed_clearvae_fp16_cleaned/resolve/main/vae-ft-mse-840000-ema-pruned_fp16.safetensors?download=true"
# - **Download LoRAs**
download_lora = "https://civitai.com/api/download/models/28907, https://huggingface.co/Leopain/color/resolve/main/Coloring_book_-_LineArt.safetensors, https://civitai.com/api/download/models/135867, https://civitai.com/api/download/models/145907, https://huggingface.co/Linaqruf/anime-detailer-xl-lora/resolve/main/anime-detailer-xl.safetensors?download=true, https://huggingface.co/Linaqruf/style-enhancer-xl-lora/resolve/main/style-enhancer-xl.safetensors?download=true, https://civitai.com/api/download/models/28609, https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-CFG-lora.safetensors?download=true, https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors?download=true"
load_diffusers_format_model = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'black-forest-labs/FLUX.1-dev',
    'John6666/blue-pencil-flux1-v021-fp8-flux',
    'John6666/wai-ani-flux-v10forfp8-fp8-flux',
    'John6666/xe-anime-flux-v04-fp8-flux',
    'John6666/lyh-anime-flux-v2a1-fp8-flux',
    'John6666/carnival-unchained-v10-fp8-flux',
    'cagliostrolab/animagine-xl-3.1',
    'John6666/epicrealism-xl-v8kiss-sdxl',
    'misri/epicrealismXL_v7FinalDestination',
    'misri/juggernautXL_juggernautX',
    'misri/zavychromaxl_v80',
    'SG161222/RealVisXL_V4.0',
    'SG161222/RealVisXL_V5.0',
    'misri/newrealityxlAllInOne_Newreality40',
    'eienmojiki/Anything-XL',
    'eienmojiki/Starry-XL-v5.2',
    'gsdf/CounterfeitXL',
    'KBlueLeaf/Kohaku-XL-Zeta',
    'John6666/silvermoon-mix-01xl-v11-sdxl',
    'WhiteAiZ/autismmixSDXL_autismmixConfetti_diffusers',
    'kitty7779/ponyDiffusionV6XL',
    'GraydientPlatformAPI/aniverse-pony',
    'John6666/ras-real-anime-screencap-v1-sdxl',
    'John6666/duchaiten-pony-xl-no-score-v60-sdxl',
    'John6666/mistoon-anime-ponyalpha-sdxl',
    'John6666/3x3x3mixxl-v2-sdxl',
    'John6666/3x3x3mixxl-3dv01-sdxl',
    'John6666/ebara-mfcg-pony-mix-v12-sdxl',
    'John6666/t-ponynai3-v51-sdxl',
    'John6666/t-ponynai3-v65-sdxl',
    'John6666/prefect-pony-xl-v3-sdxl',
    'John6666/mala-anime-mix-nsfw-pony-xl-v5-sdxl',
    'John6666/wai-real-mix-v11-sdxl',
    'John6666/wai-c-v6-sdxl',
    'John6666/iniverse-mix-xl-sfwnsfw-pony-guofeng-v43-sdxl',
    'John6666/photo-realistic-pony-v5-sdxl',
    'John6666/pony-realism-v21main-sdxl',
    'John6666/pony-realism-v22main-sdxl',
    'John6666/cyberrealistic-pony-v63-sdxl',
    'John6666/cyberrealistic-pony-v64-sdxl',
    'GraydientPlatformAPI/realcartoon-pony-diffusion',
    'John6666/nova-anime-xl-pony-v5-sdxl',
    'John6666/autismmix-sdxl-autismmix-pony-sdxl',
    'John6666/aimz-dream-real-pony-mix-v3-sdxl',
    'John6666/duchaiten-pony-real-v11fix-sdxl',
    'John6666/duchaiten-pony-real-v20-sdxl',
    'yodayo-ai/kivotos-xl-2.0',
    'yodayo-ai/holodayo-xl-2.1',
    'yodayo-ai/clandestine-xl-1.0',
    'digiplay/majicMIX_sombre_v2',
    'digiplay/majicMIX_realistic_v6',
    'digiplay/majicMIX_realistic_v7',
    'digiplay/DreamShaper_8',
    'digiplay/BeautifulArt_v1',
    'digiplay/DarkSushi2.5D_v1',
    'digiplay/darkphoenix3D_v1.1',
    'digiplay/BeenYouLiteL11_diffusers',
    'Yntec/RevAnimatedV2Rebirth',
    'youknownothing/cyberrealistic_v50',
    'youknownothing/deliberate-v6',
    'GraydientPlatformAPI/deliberate-cyber3',
    'GraydientPlatformAPI/picx-real',
    'GraydientPlatformAPI/perfectworld6',
    'emilianJR/epiCRealism',
    'votepurchase/counterfeitV30_v30',
    'votepurchase/ChilloutMix',
    'Meina/MeinaMix_V11',
    'Meina/MeinaUnreal_V5',
    'Meina/MeinaPastel_V7',
    'GraydientPlatformAPI/realcartoon3d-17',
    'GraydientPlatformAPI/realcartoon-pixar11',
    'GraydientPlatformAPI/realcartoon-real17',
]

DIFFUSERS_FORMAT_LORAS = [
    "nerijs/animation2k-flux",
    "XLabs-AI/flux-RealismLora",
]

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY")
HF_TOKEN = os.environ.get("HF_READ_TOKEN")

PREPROCESSOR_CONTROLNET = {
  "openpose": [
    "Openpose",
    "None",
  ],
  "scribble": [
    "HED",
    "PidiNet",
    "None",
  ],
  "softedge": [
    "PidiNet",
    "HED",
    "HED safe",
    "PidiNet safe",
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
    "Lineart (anime)",
    "None",
    "None (anime)",
  ],
  "lineart_anime": [
    "Lineart",
    "Lineart coarse",
    "Lineart (anime)",
    "None",
    "None (anime)",
  ],
  "shuffle": [
    "ContentShuffle",
    "None",
  ],
  "canny": [
    "Canny",
    "None",
  ],
  "mlsd": [
    "MLSD",
    "None",
  ],
  "ip2p": [
    "ip2p"
  ],
  "recolor": [
    "Recolor luminance",
    "Recolor intensity",
    "None",
  ],
  "tile": [
    "Mild Blur",
    "Moderate Blur",
    "Heavy Blur",
    "None",
  ],

}

TASK_STABLEPY = {
    'txt2img': 'txt2img',
    'img2img': 'img2img',
    'inpaint': 'inpaint',
    # 'canny T2I Adapter': 'sdxl_canny_t2i',  # NO HAVE STEP CALLBACK PARAMETERS SO NOT WORKS WITH DIFFUSERS 0.29.0
    # 'sketch  T2I Adapter': 'sdxl_sketch_t2i',
    # 'lineart  T2I Adapter': 'sdxl_lineart_t2i',
    # 'depth-midas  T2I Adapter': 'sdxl_depth-midas_t2i',
    # 'openpose  T2I Adapter': 'sdxl_openpose_t2i',
    'openpose ControlNet': 'openpose',
    'canny ControlNet': 'canny',
    'mlsd ControlNet': 'mlsd',
    'scribble ControlNet': 'scribble',
    'softedge ControlNet': 'softedge',
    'segmentation ControlNet': 'segmentation',
    'depth ControlNet': 'depth',
    'normalbae ControlNet': 'normalbae',
    'lineart ControlNet': 'lineart',
    'lineart_anime ControlNet': 'lineart_anime',
    'shuffle ControlNet': 'shuffle',
    'ip2p ControlNet': 'ip2p',
    'optical pattern ControlNet': 'pattern',
    'recolor ControlNet': 'recolor',
    'tile ControlNet': 'tile',
}

TASK_MODEL_LIST = list(TASK_STABLEPY.keys())

UPSCALER_DICT_GUI = {
    None: None,
    "Lanczos": "Lanczos",
    "Nearest": "Nearest",
    'Latent': 'Latent',
    'Latent (antialiased)': 'Latent (antialiased)',
    'Latent (bicubic)': 'Latent (bicubic)',
    'Latent (bicubic antialiased)': 'Latent (bicubic antialiased)',
    'Latent (nearest)': 'Latent (nearest)',
    'Latent (nearest-exact)': 'Latent (nearest-exact)',
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "4x-UltraSharp": "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "4x_foolhardy_Remacri": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "Remacri4xExtraSmoother": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    "AnimeSharp4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "lollypop": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    "RealisticRescaler4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    "NickelbackFS4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth"
}

UPSCALER_KEYS = list(UPSCALER_DICT_GUI.keys())


def download_things(directory, url, hf_token="", civitai_api_key=""):
    url = url.strip()

    if "drive.google.com" in url:
        original_dir = os.getcwd()
        os.chdir(directory)
        os.system(f"gdown --fuzzy {url}")
        os.chdir(original_dir)
    elif "huggingface.co" in url:
        url = url.replace("?download=true", "")
        # url = urllib.parse.quote(url, safe=':/')  # fix encoding
        if "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        user_header = f'"Authorization: Bearer {hf_token}"'
        if hf_token:
            os.system(f"aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 {url} -d {directory}  -o {url.split('/')[-1]}")
        else:
            os.system(f"aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 {url} -d {directory}  -o {url.split('/')[-1]}")
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
    valid_extensions = {'.ckpt', '.pt', '.pth', '.safetensors', '.bin'}

    for filename in os.listdir(directory_path):
        if os.path.splitext(filename)[1] in valid_extensions:
            # name_without_extension = os.path.splitext(filename)[0]
            file_path = os.path.join(directory_path, filename)
            # model_list.append((name_without_extension, file_path))
            model_list.append(file_path)
            print('\033[34mFILE: ' + file_path + '\033[0m')
    return model_list


directory_models = 'models'
os.makedirs(directory_models, exist_ok=True)
directory_loras = 'loras'
os.makedirs(directory_loras, exist_ok=True)
directory_vaes = 'vaes'
os.makedirs(directory_vaes, exist_ok=True)

# Download stuffs
for url in [url.strip() for url in download_model.split(',')]:
    if not os.path.exists(f"./models/{url.split('/')[-1]}"):
        download_things(directory_models, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in download_vae.split(',')]:
    if not os.path.exists(f"./vaes/{url.split('/')[-1]}"):
        download_things(directory_vaes, url, HF_TOKEN, CIVITAI_API_KEY)
for url in [url.strip() for url in download_lora.split(',')]:
    if not os.path.exists(f"./loras/{url.split('/')[-1]}"):
        download_things(directory_loras, url, HF_TOKEN, CIVITAI_API_KEY)

# Download Embeddings
directory_embeds = 'embedings'
os.makedirs(directory_embeds, exist_ok=True)
download_embeds = [
    'https://huggingface.co/datasets/Nerfgun3/bad_prompt/blob/main/bad_prompt_version2.pt',
    'https://huggingface.co/embed/negative/resolve/main/EasyNegativeV2.safetensors',
    'https://huggingface.co/embed/negative/resolve/main/bad-hands-5.pt',
    ]

for url_embed in download_embeds:
    if not os.path.exists(f"./embedings/{url_embed.split('/')[-1]}"):
        download_things(directory_embeds, url_embed, HF_TOKEN, CIVITAI_API_KEY)

# Build list models
embed_list = get_model_list(directory_embeds)
model_list = get_model_list(directory_models)
model_list = load_diffusers_format_model + model_list
lora_model_list = get_model_list(directory_loras)
lora_model_list.insert(0, "None")
lora_model_list = lora_model_list + DIFFUSERS_FORMAT_LORAS
vae_model_list = get_model_list(directory_vaes)
vae_model_list.insert(0, "None")

print('\033[33m🏁 Download and listing of valid models completed.\033[0m')

#######################
# GUI
#######################
import gradio as gr
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

msg_inc_vae = (
    "Use the right VAE for your model to maintain image quality. The wrong"
    " VAE can lead to poor results, like blurriness in the generated images."
)

SDXL_TASK = [k for k, v in TASK_STABLEPY.items() if v in SDXL_TASKS]
SD_TASK = [k for k, v in TASK_STABLEPY.items() if v in SD15_TASKS]
FLUX_TASK = list(TASK_STABLEPY.keys())[:3] + [k for k, v in TASK_STABLEPY.items() if v in FLUX_CN_UNION_MODES.keys()]

MODEL_TYPE_TASK = {
    "SD 1.5": SD_TASK,
    "SDXL": SDXL_TASK,
    "FLUX": FLUX_TASK,
}

MODEL_TYPE_CLASS = {
    "diffusers:StableDiffusionPipeline": "SD 1.5",
    "diffusers:StableDiffusionXLPipeline": "SDXL",
    "diffusers:FluxPipeline": "FLUX",
}

POST_PROCESSING_SAMPLER = ["Use same sampler"] + scheduler_names[:-2]

CSS = """
.contain { display: flex; flex-direction: column; }
#component-0 { height: 100%; }
#gallery { flex-grow: 1; }
"""

SUBTITLE_GUI = (
    "### This demo uses [diffusers](https://github.com/huggingface/diffusers)"
    " to perform different tasks in image generation."
)


def extract_parameters(input_string):
    parameters = {}
    input_string = input_string.replace("\n", "")

    if "Negative prompt:" not in input_string:
        if "Steps:" in input_string:
            input_string = input_string.replace("Steps:", "Negative prompt: Steps:")
        else:
            print("Invalid metadata")
            parameters["prompt"] = input_string
            return parameters

    parm = input_string.split("Negative prompt:")
    parameters["prompt"] = parm[0].strip()
    if "Steps:" not in parm[1]:
        print("Steps not detected")
        parameters["neg_prompt"] = parm[1].strip()
        return parameters
    parm = parm[1].split("Steps:")
    parameters["neg_prompt"] = parm[0].strip()
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


def get_my_lora(link_url):
    for url in [url.strip() for url in link_url.split(',')]:
        if not os.path.exists(f"./loras/{url.split('/')[-1]}"):
            download_things(directory_loras, url, HF_TOKEN, CIVITAI_API_KEY)
    new_lora_model_list = get_model_list(directory_loras)
    new_lora_model_list.insert(0, "None")
    new_lora_model_list = new_lora_model_list + DIFFUSERS_FORMAT_LORAS

    return gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ), gr.update(
        choices=new_lora_model_list
    ),


def info_html(json_data, title, subtitle):
    return f"""
        <div style='padding: 0; border-radius: 10px;'>
            <p style='margin: 0; font-weight: bold;'>{title}</p>
            <details>
                <summary>Details</summary>
                <p style='margin: 0; font-weight: bold;'>{subtitle}</p>
            </details>
        </div>
        """


def get_model_type(repo_id: str):
    api = HfApi(token=os.environ.get("HF_TOKEN"))  # if use private or gated model
    default = "SD 1.5"
    try:
        model = api.model_info(repo_id=repo_id, timeout=5.0)
        tags = model.tags
        for tag in tags:
            if tag in MODEL_TYPE_CLASS.keys(): return MODEL_TYPE_CLASS.get(tag, default)
    except Exception:
        return default
    return default


class GuiSD:
    def __init__(self, stream=True):
        self.model = None

        print("Loading model...")
        self.model = Model_Diffusers(
            base_model_id="Lykon/dreamshaper-8",
            task_name="txt2img",
            vae_model=None,
            type_model_precision=torch.float16,
            retain_task_model_in_cache=False,
            device="cpu",
        )
        self.model.load_beta_styles()

    def load_new_model(self, model_name, vae_model, task, progress=gr.Progress(track_tqdm=True)):

        yield f"Loading model: {model_name}"

        vae_model = vae_model if vae_model != "None" else None
        model_type = get_model_type(model_name)

        if vae_model:
            vae_type = "SDXL" if "sdxl" in vae_model.lower() else "SD 1.5"
            if model_type != vae_type:
                gr.Warning(msg_inc_vae)

        self.model.device = torch.device("cpu")
        dtype_model = torch.bfloat16 if model_type == "FLUX" else torch.float16

        self.model.load_pipe(
            model_name,
            task_name=TASK_STABLEPY[task],
            vae_model=vae_model,
            type_model_precision=dtype_model,
            retain_task_model_in_cache=False,
        )

        yield f"Model loaded: {model_name}"

    # @spaces.GPU(duration=59)
    @torch.inference_mode()
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
        retain_task_cache_gui,
        image_ip1,
        mask_ip1,
        model_ip1,
        mode_ip1,
        scale_ip1,
        image_ip2,
        mask_ip2,
        model_ip2,
        mode_ip2,
        scale_ip2,
        pag_scale,
    ):

        vae_model = vae_model if vae_model != "None" else None
        loras_list = [lora1, lora2, lora3, lora4, lora5]
        vae_msg = f"VAE: {vae_model}" if vae_model else ""
        msg_lora = ""

        print("Config model:", model_name, vae_model, loras_list)

        task = TASK_STABLEPY[task]

        params_ip_img = []
        params_ip_msk = []
        params_ip_model = []
        params_ip_mode = []
        params_ip_scale = []

        all_adapters = [
            (image_ip1, mask_ip1, model_ip1, mode_ip1, scale_ip1),
            (image_ip2, mask_ip2, model_ip2, mode_ip2, scale_ip2),
        ]

        for imgip, mskip, modelip, modeip, scaleip in all_adapters:
            if imgip:
                params_ip_img.append(imgip)
                if mskip:
                    params_ip_msk.append(mskip)
                params_ip_model.append(modelip)
                params_ip_mode.append(modeip)
                params_ip_scale.append(scaleip)

        self.model.stream_config(concurrency=5, latent_resize_by=1, vae_decoding=False)

        if task != "txt2img" and not image_control:
            raise ValueError("No control image found: To use this function, you have to upload an image in 'Image ControlNet/Inpaint/Img2img'")

        if task == "inpaint" and not image_mask:
            raise ValueError("No mask image found: Specify one in 'Image Mask'")

        if upscaler_model_path in UPSCALER_KEYS[:9]:
            upscaler_model = upscaler_model_path
        else:
            directory_upscalers = 'upscalers'
            os.makedirs(directory_upscalers, exist_ok=True)

            url_upscaler = UPSCALER_DICT_GUI[upscaler_model_path]

            if not os.path.exists(f"./upscalers/{url_upscaler.split('/')[-1]}"):
                download_things(directory_upscalers, url_upscaler, HF_TOKEN)

            upscaler_model = f"./upscalers/{url_upscaler.split('/')[-1]}"

        logging.getLogger("ultralytics").setLevel(logging.INFO if adetailer_verbose else logging.ERROR)

        adetailer_params_A = {
            "face_detector_ad": face_detector_ad_a,
            "person_detector_ad": person_detector_ad_a,
            "hand_detector_ad": hand_detector_ad_a,
            "prompt": prompt_ad_a,
            "negative_prompt": negative_prompt_ad_a,
            "strength": strength_ad_a,
            # "image_list_task" : None,
            "mask_dilation": mask_dilation_a,
            "mask_blur": mask_blur_a,
            "mask_padding": mask_padding_a,
            "inpaint_only": adetailer_inpaint_only,
            "sampler": adetailer_sampler,
        }

        adetailer_params_B = {
            "face_detector_ad": face_detector_ad_b,
            "person_detector_ad": person_detector_ad_b,
            "hand_detector_ad": hand_detector_ad_b,
            "prompt": prompt_ad_b,
            "negative_prompt": negative_prompt_ad_b,
            "strength": strength_ad_b,
            # "image_list_task" : None,
            "mask_dilation": mask_dilation_b,
            "mask_blur": mask_blur_b,
            "mask_padding": mask_padding_b,
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
            "pag_scale": float(pag_scale),
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
            "hires_after_adetailer": hires_after_adetailer,
            "ip_adapter_image": params_ip_img,
            "ip_adapter_mask": params_ip_msk,
            "ip_adapter_model": params_ip_model,
            "ip_adapter_mode": params_ip_mode,
            "ip_adapter_scale": params_ip_scale,
        }

        self.model.device = torch.device("cuda:0")
        if hasattr(self.model.pipe, "transformer") and loras_list != ["None"] * 5:
            self.model.pipe.transformer.to(self.model.device)
            print("transformer to cuda")

        info_state = "PROCESSING "
        for img, seed, image_path, metadata in self.model(**pipe_params):
            info_state += ">"
            if image_path:
                info_state = f"COMPLETE. Seeds: {str(seed)}"
                if vae_msg:
                    info_state = info_state + "<br>" + vae_msg

                for status, lora in zip(self.model.lora_status, self.model.lora_memory):
                    if status:
                        msg_lora += f"<br>Loaded: {lora}"
                    elif status is not None:
                        msg_lora += f"<br>Error with: {lora}"

                if msg_lora:
                    info_state += msg_lora

                info_state = info_state + "<br>" + "GENERATION DATA:<br>" + metadata[0].replace("\n", "<br>") + "<br>-------<br>"

                download_links = "<br>".join(
                    [
                        f'<a href="{path.replace("/images/", "/file=/home/user/app/images/")}" download="{os.path.basename(path)}">Download Image {i + 1}</a>'
                        for i, path in enumerate(image_path)
                    ]
                )
                if save_generated_images:
                    info_state += f"<br>{download_links}"

            yield img, info_state


def update_task_options(model_name, task_name):
    new_choices = MODEL_TYPE_TASK[get_model_type(model_name)]

    if task_name not in new_choices:
        task_name = "txt2img"

    return gr.update(value=task_name, choices=new_choices)


def dynamic_gpu_duration(func, duration, *args):

    @spaces.GPU(duration=duration)
    def wrapped_func():
        yield from func(*args)

    return wrapped_func()


@spaces.GPU
def dummy_gpu():
    return None


def sd_gen_generate_pipeline(*args):

    gpu_duration_arg = int(args[-1]) if args[-1] else 59
    verbose_arg = int(args[-2])
    load_lora_cpu = args[-3]
    generation_args = args[:-3]
    lora_list = [
        None if item == "None" else item
        for item in [args[7], args[9], args[11], args[13], args[15]]
    ]
    lora_status = [None] * 5

    msg_load_lora = "Updating LoRAs in GPU..."
    if load_lora_cpu:
        msg_load_lora = "Updating LoRAs in CPU (Slow but saves GPU usage)..."

    if lora_list != sd_gen.model.lora_memory and lora_list != [None] * 5:
        yield None, msg_load_lora

    # Load lora in CPU
    if load_lora_cpu:
        lora_status = sd_gen.model.lora_merge(
            lora_A=lora_list[0], lora_scale_A=args[8],
            lora_B=lora_list[1], lora_scale_B=args[10],
            lora_C=lora_list[2], lora_scale_C=args[12],
            lora_D=lora_list[3], lora_scale_D=args[14],
            lora_E=lora_list[4], lora_scale_E=args[16],
        )
        print(lora_status)

    if verbose_arg:
        for status, lora in zip(lora_status, lora_list):
            if status:
                gr.Info(f"LoRA loaded in CPU: {lora}")
            elif status is not None:
                gr.Warning(f"Failed to load LoRA: {lora}")

        if lora_status == [None] * 5 and sd_gen.model.lora_memory != [None] * 5 and load_lora_cpu:
            lora_cache_msg = ", ".join(
                str(x) for x in sd_gen.model.lora_memory if x is not None
            )
            gr.Info(f"LoRAs in cache: {lora_cache_msg}")

        msg_request = f"Requesting {gpu_duration_arg}s. of GPU time"
        gr.Info(msg_request)
        print(msg_request)

    # yield from sd_gen.generate_pipeline(*generation_args)

    start_time = time.time()

    yield from dynamic_gpu_duration(
        sd_gen.generate_pipeline,
        gpu_duration_arg,
        *generation_args,
    )

    end_time = time.time()

    if verbose_arg:
        execution_time = end_time - start_time
        msg_task_complete = (
            f"GPU task complete in: {round(execution_time, 0) + 1} seconds"
        )
        gr.Info(msg_task_complete)
        print(msg_task_complete)


def extract_exif_data(image):
    if image is None: return ""

    try:
        metadata_keys = ['parameters', 'metadata', 'prompt', 'Comment']

        for key in metadata_keys:
            if key in image.info:
                return image.info[key]

        return str(image.info)

    except Exception as e:
        return f"Error extracting metadata: {str(e)}"


@spaces.GPU(duration=20)
def esrgan_upscale(image, upscaler_name, upscaler_size):
    if image is None: return None

    from stablepy.diffusers_vanilla.utils import save_pil_image_with_metadata
    from stablepy import UpscalerESRGAN

    exif_image = extract_exif_data(image)

    url_upscaler = UPSCALER_DICT_GUI[upscaler_name]
    directory_upscalers = 'upscalers'
    os.makedirs(directory_upscalers, exist_ok=True)
    if not os.path.exists(f"./upscalers/{url_upscaler.split('/')[-1]}"):
        download_things(directory_upscalers, url_upscaler, HF_TOKEN)

    scaler_beta = UpscalerESRGAN(0, 0)
    image_up = scaler_beta.upscale(image, upscaler_size, f"./upscalers/{url_upscaler.split('/')[-1]}")

    image_path = save_pil_image_with_metadata(image_up, f'{os.getcwd()}/up_images', exif_image)

    return image_path


dynamic_gpu_duration.zerogpu = True
sd_gen_generate_pipeline.zerogpu = True
sd_gen = GuiSD()

with gr.Blocks(theme="NoCrypt/miku", css=CSS) as app:
    gr.Markdown("# 🧩 DiffuseCraft")
    gr.Markdown(SUBTITLE_GUI)
    with gr.Tab("Generation"):
        with gr.Row():

            with gr.Column(scale=2):

                task_gui = gr.Dropdown(label="Task", choices=SDXL_TASK, value=TASK_MODEL_LIST[0])
                model_name_gui = gr.Dropdown(label="Model", choices=model_list, value=model_list[0], allow_custom_value=True)
                prompt_gui = gr.Textbox(lines=5, placeholder="Enter prompt", label="Prompt")
                neg_prompt_gui = gr.Textbox(lines=3, placeholder="Enter Neg prompt", label="Negative prompt")
                with gr.Row(equal_height=False):
                    set_params_gui = gr.Button(value="↙️", variant="secondary", size="sm")
                    clear_prompt_gui = gr.Button(value="🗑️", variant="secondary", size="sm")
                    set_random_seed = gr.Button(value="🎲", variant="secondary", size="sm")
                generate_button = gr.Button(value="GENERATE IMAGE", variant="primary")

                model_name_gui.change(
                    update_task_options,
                    [model_name_gui, task_gui],
                    [task_gui],
                )

                load_model_gui = gr.HTML()

                result_images = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                    columns=[2],
                    rows=[2],
                    object_fit="contain",
                    # height="auto",
                    interactive=False,
                    preview=False,
                    selected_index=50,
                )

                actual_task_info = gr.HTML()

                with gr.Row(equal_height=False, variant="default"):
                    gpu_duration_gui = gr.Number(minimum=5, maximum=240, value=59, show_label=False, container=False, info="GPU time duration (seconds)")
                    with gr.Column():
                        verbose_info_gui = gr.Checkbox(value=False, container=False, label="Status info")
                        load_lora_cpu_gui = gr.Checkbox(value=False, container=False, label="Load LoRAs on CPU (Save GPU time)")

            with gr.Column(scale=1):
                steps_gui = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Steps")
                cfg_gui = gr.Slider(minimum=0, maximum=30, step=0.5, value=7., label="CFG")
                sampler_gui = gr.Dropdown(label="Sampler", choices=scheduler_names, value="Euler a")
                img_width_gui = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Width")
                img_height_gui = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="Img Height")
                seed_gui = gr.Number(minimum=-1, maximum=9999999999, value=-1, label="Seed")
                pag_scale_gui = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, label="PAG Scale")
                with gr.Row():
                    clip_skip_gui = gr.Checkbox(value=True, label="Layer 2 Clip Skip")
                    free_u_gui = gr.Checkbox(value=False, label="FreeU")

                with gr.Row(equal_height=False):

                    def run_set_params_gui(base_prompt, name_model):
                        valid_receptors = {  # default values
                            "prompt": gr.update(value=base_prompt),
                            "neg_prompt": gr.update(value=""),
                            "Steps": gr.update(value=30),
                            "width": gr.update(value=1024),
                            "height": gr.update(value=1024),
                            "Seed": gr.update(value=-1),
                            "Sampler": gr.update(value="Euler a"),
                            "scale": gr.update(value=7.),  # cfg
                            "skip": gr.update(value=True),
                            "Model": gr.update(value=name_model),
                        }
                        valid_keys = list(valid_receptors.keys())

                        parameters = extract_parameters(base_prompt)

                        for key, val in parameters.items():
                            # print(val)
                            if key in valid_keys:
                                try:
                                    if key == "Sampler":
                                        if val not in scheduler_names:
                                            continue
                                    elif key == "skip":
                                        if "," in str(val):
                                            val = val.replace(",", "")
                                        if int(val) >= 2:
                                            val = True
                                    if key == "prompt":
                                        if ">" in val and "<" in val:
                                            val = re.sub(r'<[^>]+>', '', val)
                                            print("Removed LoRA written in the prompt")
                                    if key in ["prompt", "neg_prompt"]:
                                        val = re.sub(r'\s+', ' ', re.sub(r',+', ',', val)).strip()
                                    if key in ["Steps", "width", "height", "Seed"]:
                                        val = int(val)
                                    if key == "scale":
                                        val = float(val)
                                    if key == "Model":
                                        filtered_models = [m for m in model_list if val in m]
                                        if filtered_models:
                                            val = filtered_models[0]
                                        else:
                                            val = name_model
                                    if key == "Seed":
                                        continue
                                    valid_receptors[key] = gr.update(value=val)
                                    # print(val, type(val))
                                    # print(valid_receptors)
                                except Exception as e:
                                    print(str(e))
                        return [value for value in valid_receptors.values()]

                    set_params_gui.click(
                        run_set_params_gui, [prompt_gui, model_name_gui], [
                            prompt_gui,
                            neg_prompt_gui,
                            steps_gui,
                            img_width_gui,
                            img_height_gui,
                            seed_gui,
                            sampler_gui,
                            cfg_gui,
                            clip_skip_gui,
                            model_name_gui,
                        ],
                    )

                    def run_clear_prompt_gui():
                        return gr.update(value=""), gr.update(value="")
                    clear_prompt_gui.click(
                        run_clear_prompt_gui, [], [prompt_gui, neg_prompt_gui]
                    )

                    def run_set_random_seed():
                        return -1
                    set_random_seed.click(
                        run_set_random_seed, [], seed_gui
                    )

                num_images_gui = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Images")
                prompt_s_options = [
                    ("Compel format: (word)weight", "Compel"),
                    ("Classic format: (word:weight)", "Classic"),
                    ("Classic-original format: (word:weight)", "Classic-original"),
                    ("Classic-no_norm format: (word:weight)", "Classic-no_norm"),
                    ("Classic-ignore", "Classic-ignore"),
                    ("None", "None"),
                ]
                prompt_syntax_gui = gr.Dropdown(label="Prompt Syntax", choices=prompt_s_options, value=prompt_s_options[1][1])
                vae_model_gui = gr.Dropdown(label="VAE Model", choices=vae_model_list, value=vae_model_list[0])

                with gr.Accordion("Hires fix", open=False, visible=True):

                    upscaler_model_path_gui = gr.Dropdown(label="Upscaler", choices=UPSCALER_KEYS, value=UPSCALER_KEYS[0])
                    upscaler_increases_size_gui = gr.Slider(minimum=1.1, maximum=4., step=0.1, value=1.2, label="Upscale by")
                    esrgan_tile_gui = gr.Slider(minimum=0, value=0, maximum=500, step=1, label="ESRGAN Tile")
                    esrgan_tile_overlap_gui = gr.Slider(minimum=1, maximum=200, step=1, value=8, label="ESRGAN Tile Overlap")
                    hires_steps_gui = gr.Slider(minimum=0, value=30, maximum=100, step=1, label="Hires Steps")
                    hires_denoising_strength_gui = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.55, label="Hires Denoising Strength")
                    hires_sampler_gui = gr.Dropdown(label="Hires Sampler", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])
                    hires_prompt_gui = gr.Textbox(label="Hires Prompt", placeholder="Main prompt will be use", lines=3)
                    hires_negative_prompt_gui = gr.Textbox(label="Hires Negative Prompt", placeholder="Main negative prompt will be use", lines=3)

                with gr.Accordion("LoRA", open=False, visible=True):

                    def lora_dropdown(label):
                        return gr.Dropdown(label=label, choices=lora_model_list, value="None", allow_custom_value=True)

                    def lora_scale_slider(label):
                        return gr.Slider(minimum=-2, maximum=2, step=0.01, value=0.33, label=label)

                    lora1_gui = lora_dropdown("Lora1")
                    lora_scale_1_gui = lora_scale_slider("Lora Scale 1")
                    lora2_gui = lora_dropdown("Lora2")
                    lora_scale_2_gui = lora_scale_slider("Lora Scale 2")
                    lora3_gui = lora_dropdown("Lora3")
                    lora_scale_3_gui = lora_scale_slider("Lora Scale 3")
                    lora4_gui = lora_dropdown("Lora4")
                    lora_scale_4_gui = lora_scale_slider("Lora Scale 4")
                    lora5_gui = lora_dropdown("Lora5")
                    lora_scale_5_gui = lora_scale_slider("Lora Scale 5")

                    with gr.Accordion("From URL", open=False, visible=True):
                        text_lora = gr.Textbox(label="LoRA URL", placeholder="https://civitai.com/api/download/models/28907", lines=1)
                        button_lora = gr.Button("Get and update lists of LoRAs")
                        button_lora.click(
                            get_my_lora,
                            [text_lora],
                            [lora1_gui, lora2_gui, lora3_gui, lora4_gui, lora5_gui]
                        )

                with gr.Accordion("IP-Adapter", open=False, visible=True):

                    IP_MODELS = sorted(list(set(IP_ADAPTERS_SD + IP_ADAPTERS_SDXL)))
                    MODE_IP_OPTIONS = ["original", "style", "layout", "style+layout"]

                    with gr.Accordion("IP-Adapter 1", open=False, visible=True):
                        image_ip1 = gr.Image(label="IP Image", type="filepath")
                        mask_ip1 = gr.Image(label="IP Mask", type="filepath")
                        model_ip1 = gr.Dropdown(value="plus_face", label="Model", choices=IP_MODELS)
                        mode_ip1 = gr.Dropdown(value="original", label="Mode", choices=MODE_IP_OPTIONS)
                        scale_ip1 = gr.Slider(minimum=0., maximum=2., step=0.01, value=0.7, label="Scale")
                    with gr.Accordion("IP-Adapter 2", open=False, visible=True):
                        image_ip2 = gr.Image(label="IP Image", type="filepath")
                        mask_ip2 = gr.Image(label="IP Mask (optional)", type="filepath")
                        model_ip2 = gr.Dropdown(value="base", label="Model", choices=IP_MODELS)
                        mode_ip2 = gr.Dropdown(value="style", label="Mode", choices=MODE_IP_OPTIONS)
                        scale_ip2 = gr.Slider(minimum=0., maximum=2., step=0.01, value=0.7, label="Scale")

                with gr.Accordion("ControlNet / Img2img / Inpaint", open=False, visible=True):
                    image_control = gr.Image(label="Image ControlNet/Inpaint/Img2img", type="filepath")
                    image_mask_gui = gr.Image(label="Image Mask", type="filepath")
                    strength_gui = gr.Slider(
                        minimum=0.01, maximum=1.0, step=0.01, value=0.55, label="Strength",
                        info="This option adjusts the level of changes for img2img and inpainting."
                    )
                    image_resolution_gui = gr.Slider(minimum=64, maximum=2048, step=64, value=1024, label="Image Resolution")
                    preprocessor_name_gui = gr.Dropdown(label="Preprocessor Name", choices=PREPROCESSOR_CONTROLNET["canny"])

                    def change_preprocessor_choices(task):
                        task = TASK_STABLEPY[task]
                        if task in PREPROCESSOR_CONTROLNET.keys():
                            choices_task = PREPROCESSOR_CONTROLNET[task]
                        else:
                            choices_task = PREPROCESSOR_CONTROLNET["canny"]
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

                with gr.Accordion("T2I adapter", open=False, visible=False):
                    t2i_adapter_preprocessor_gui = gr.Checkbox(value=True, label="T2i Adapter Preprocessor")
                    adapter_conditioning_scale_gui = gr.Slider(minimum=0, maximum=5., step=0.1, value=1, label="Adapter Conditioning Scale")
                    adapter_conditioning_factor_gui = gr.Slider(minimum=0, maximum=1., step=0.01, value=0.55, label="Adapter Conditioning Factor (%)")

                with gr.Accordion("Styles", open=False, visible=True):

                    try:
                        style_names_found = sd_gen.model.STYLE_NAMES
                    except Exception:
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

                with gr.Accordion("Detailfix", open=False, visible=True):

                    # Adetailer Inpaint Only
                    adetailer_inpaint_only_gui = gr.Checkbox(label="Inpaint only", value=True)

                    # Adetailer Verbose
                    adetailer_verbose_gui = gr.Checkbox(label="Verbose", value=False)

                    # Adetailer Sampler
                    adetailer_sampler_gui = gr.Dropdown(label="Adetailer sampler:", choices=POST_PROCESSING_SAMPLER, value=POST_PROCESSING_SAMPLER[0])

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

                with gr.Accordion("Other settings", open=False, visible=True):
                    save_generated_images_gui = gr.Checkbox(value=True, label="Create a download link for the images")
                    hires_before_adetailer_gui = gr.Checkbox(value=False, label="Hires Before Adetailer")
                    hires_after_adetailer_gui = gr.Checkbox(value=True, label="Hires After Adetailer")
                    generator_in_cpu_gui = gr.Checkbox(value=False, label="Generator in CPU")

                with gr.Accordion("More settings", open=False, visible=False):
                    loop_generation_gui = gr.Slider(minimum=1, value=1, label="Loop Generation")
                    retain_task_cache_gui = gr.Checkbox(value=False, label="Retain task model in cache")
                    leave_progress_bar_gui = gr.Checkbox(value=True, label="Leave Progress Bar")
                    disable_progress_bar_gui = gr.Checkbox(value=False, label="Disable Progress Bar")
                    display_images_gui = gr.Checkbox(value=True, label="Display Images")
                    image_previews_gui = gr.Checkbox(value=True, label="Image Previews")
                    image_storage_location_gui = gr.Textbox(value="./images", label="Image Storage Location")
                    retain_compel_previous_load_gui = gr.Checkbox(value=False, label="Retain Compel Previous Load")
                    retain_detailfix_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Detailfix Model Previous Load")
                    retain_hires_model_previous_load_gui = gr.Checkbox(value=False, label="Retain Hires Model Previous Load")
                    xformers_memory_efficient_attention_gui = gr.Checkbox(value=False, label="Xformers Memory Efficient Attention")

        with gr.Accordion("Examples and help", open=False, visible=True):
            gr.Markdown(
                """### Help:
                - The current space runs on a ZERO GPU which is assigned for approximately 60 seconds; Therefore, if you submit expensive tasks, the operation may be canceled upon reaching the maximum allowed time with 'GPU TASK ABORTED'.
                - Distorted or strange images often result from high prompt weights, so it's best to use low weights and scales, and consider using Classic variants like 'Classic-original'.
                - For better results with Pony Diffusion, try using sampler DPM++ 1s or DPM2 with Compel or Classic prompt weights.
                """
            )
            gr.Markdown(
                """### The following examples perform specific tasks:
                1. Generation with SDXL and upscale
                2. Generation with FLUX dev
                3. ControlNet Canny SDXL
                4. Optical pattern (Optical illusion) SDXL
                5. Convert an image to a coloring drawing
                6. ControlNet OpenPose SD 1.5 and Latent upscale

                - Different tasks can be performed, such as img2img or using the IP adapter, to preserve a person's appearance or a specific style based on an image.
                """
            )
            gr.Examples(
                examples=[
                    [
                        "1girl, souryuu asuka langley, neon genesis evangelion, rebuild of evangelion, lance of longinus, cat hat, plugsuit, pilot suit, red bodysuit, sitting, crossed legs, black eye patch, throne, looking down, from bottom, looking at viewer, outdoors, (masterpiece), (best quality), (ultra-detailed), very aesthetic, illustration, disheveled hair, perfect composition, moist skin, intricate details",
                        "nfsw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, unfinished, very displeasing, oldest, early, chromatic aberration, artistic error, scan, abstract",
                        28,
                        7.0,
                        -1,
                        "None",
                        0.33,
                        "Euler a",
                        1152,
                        896,
                        "cagliostrolab/animagine-xl-3.1",
                        "txt2img",
                        "image.webp",  # img conttol
                        1024,  # img resolution
                        0.35,  # strength
                        1.0,  # cn scale
                        0.0,  # cn start
                        1.0,  # cn end
                        "Classic",
                        "Nearest",
                        45,
                        False,
                    ],
                    [
                        "a digital illustration of a movie poster titled 'Finding Emo', finding nemo parody poster, featuring a depressed cartoon clownfish with black emo hair, eyeliner, and piercings, bored expression, swimming in a dark underwater scene, in the background, movie title in a dripping, grungy font, moody blue and purple color palette",
                        "",
                        24,
                        3.5,
                        -1,
                        "None",
                        0.33,
                        "Euler a",
                        1152,
                        896,
                        "black-forest-labs/FLUX.1-dev",
                        "txt2img",
                        None,  # img conttol
                        1024,  # img resolution
                        0.35,  # strength
                        1.0,  # cn scale
                        0.0,  # cn start
                        1.0,  # cn end
                        "Classic",
                        None,
                        70,
                        True,
                    ],
                    [
                        "((masterpiece)), best quality, blonde disco girl, detailed face, realistic face, realistic hair, dynamic pose, pink pvc, intergalactic disco background, pastel lights, dynamic contrast, airbrush, fine detail, 70s vibe, midriff",
                        "(worst quality:1.2), (bad quality:1.2), (poor quality:1.2), (missing fingers:1.2), bad-artist-anime, bad-artist, bad-picture-chill-75v",
                        48,
                        3.5,
                        -1,
                        "None",
                        0.33,
                        "DPM++ 2M SDE Lu",
                        1024,
                        1024,
                        "misri/epicrealismXL_v7FinalDestination",
                        "canny ControlNet",
                        "image.webp",  # img conttol
                        1024,  # img resolution
                        0.35,  # strength
                        1.0,  # cn scale
                        0.0,  # cn start
                        1.0,  # cn end
                        "Classic",
                        None,
                        44,
                        False,
                    ],
                    [
                        "cinematic scenery old city ruins",
                        "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), (illustration, 3d, 2d, painting, cartoons, sketch, blurry, film grain, noise), (low quality, worst quality:1.2)",
                        50,
                        4.0,
                        -1,
                        "None",
                        0.33,
                        "Euler a",
                        1024,
                        1024,
                        "misri/juggernautXL_juggernautX",
                        "optical pattern ControlNet",
                        "spiral_no_transparent.png",  # img conttol
                        1024,  # img resolution
                        0.35,  # strength
                        1.0,  # cn scale
                        0.05,  # cn start
                        0.75,  # cn end
                        "Classic",
                        None,
                        35,
                        False,
                    ],
                    [
                        "black and white, line art, coloring drawing, clean line art, black strokes, no background, white, black, free lines, black scribbles, on paper, A blend of comic book art and lineart full of black and white color, masterpiece, high-resolution, trending on Pixiv fan box, palette knife, brush strokes, two-dimensional, planar vector, T-shirt design, stickers, and T-shirt design, vector art, fantasy art, Adobe Illustrator, hand-painted, digital painting, low polygon, soft lighting, aerial view, isometric style, retro aesthetics, 8K resolution, black sketch lines, monochrome, invert color",
                        "color, red, green, yellow, colored, duplicate, blurry, abstract, disfigured, deformed, animated, toy, figure, framed, 3d, bad art, poorly drawn, extra limbs, close up, b&w, weird colors, blurry, watermark, blur haze, 2 heads, long neck, watermark, elongated body, cropped image, out of frame, draft, deformed hands, twisted fingers, double image, malformed hands, multiple heads, extra limb, ugly, poorly drawn hands, missing limb, cut-off, over satured, grain, lowères, bad anatomy, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, out of focus, long body, disgusting, extra fingers, groos proportions, missing arms, mutated hands, cloned face, missing legs, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, bluelish, blue",
                        20,
                        4.0,
                        -1,
                        "loras/Coloring_book_-_LineArt.safetensors",
                        1.0,
                        "DPM++ 2M SDE Karras",
                        1024,
                        1024,
                        "cagliostrolab/animagine-xl-3.1",
                        "lineart ControlNet",
                        "color_image.png",  # img conttol
                        896,  # img resolution
                        0.35,  # strength
                        1.0,  # cn scale
                        0.0,  # cn start
                        1.0,  # cn end
                        "Compel",
                        None,
                        35,
                        False,
                    ],
                    [
                        "1girl,face,curly hair,red hair,white background,",
                        "(worst quality:2),(low quality:2),(normal quality:2),lowres,watermark,",
                        38,
                        5.0,
                        -1,
                        "None",
                        0.33,
                        "DPM++ 2M SDE Karras",
                        512,
                        512,
                        "digiplay/majicMIX_realistic_v7",
                        "openpose ControlNet",
                        "image.webp",  # img conttol
                        1024,  # img resolution
                        0.35,  # strength
                        1.0,  # cn scale
                        0.0,  # cn start
                        0.9,  # cn end
                        "Compel",
                        "Latent (antialiased)",
                        46,
                        False,
                    ],
                ],
                fn=sd_gen.generate_pipeline,
                inputs=[
                    prompt_gui,
                    neg_prompt_gui,
                    steps_gui,
                    cfg_gui,
                    seed_gui,
                    lora1_gui,
                    lora_scale_1_gui,
                    sampler_gui,
                    img_height_gui,
                    img_width_gui,
                    model_name_gui,
                    task_gui,
                    image_control,
                    image_resolution_gui,
                    strength_gui,
                    control_net_output_scaling_gui,
                    control_net_start_threshold_gui,
                    control_net_stop_threshold_gui,
                    prompt_syntax_gui,
                    upscaler_model_path_gui,
                    gpu_duration_gui,
                    load_lora_cpu_gui,
                ],
                outputs=[result_images, actual_task_info],
                cache_examples=False,
            )
            gr.Markdown(
                """### Resources
                - John6666's space has some great features you might find helpful [link](https://huggingface.co/spaces/John6666/DiffuseCraftMod).
                - You can also try the image generator in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/SD_diffusers_interactive).
                """
            )

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
                image_base = gr.ImageEditor(
                    sources=["upload", "clipboard"],
                    # crop_size="1:1",
                    # enable crop (or disable it)
                    # transforms=["crop"],
                    brush=gr.Brush(
                      default_size="16",  # or leave it as 'auto'
                      color_mode="fixed",  # 'fixed' hides the user swatches and colorpicker, 'defaults' shows it
                      # default_color="black", # html names are supported
                      colors=[
                        "rgba(0, 0, 0, 1)",  # rgb(a)
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

    with gr.Tab("PNG Info"):

        with gr.Row():
            with gr.Column():
                image_metadata = gr.Image(label="Image with metadata", type="pil", sources=["upload"])

            with gr.Column():
                result_metadata = gr.Textbox(label="Metadata", show_label=True, show_copy_button=True, interactive=False, container=True, max_lines=99)

                image_metadata.change(
                    fn=extract_exif_data,
                    inputs=[image_metadata],
                    outputs=[result_metadata],
                )

    with gr.Tab("Upscaler"):

        with gr.Row():
            with gr.Column():
                image_up_tab = gr.Image(label="Image", type="pil", sources=["upload"])
                upscaler_tab = gr.Dropdown(label="Upscaler", choices=UPSCALER_KEYS[9:], value=UPSCALER_KEYS[11])
                upscaler_size_tab = gr.Slider(minimum=1., maximum=4., step=0.1, value=1.1, label="Upscale by")
                generate_button_up_tab = gr.Button(value="START UPSCALE", variant="primary")

            with gr.Column():
                result_up_tab = gr.Image(label="Result", type="pil", interactive=False, format="png")

                generate_button_up_tab.click(
                    fn=esrgan_upscale,
                    inputs=[image_up_tab, upscaler_tab, upscaler_size_tab],
                    outputs=[result_up_tab],
                )

    generate_button.click(
        fn=sd_gen.load_new_model,
        inputs=[
            model_name_gui,
            vae_model_gui,
            task_gui
        ],
        outputs=[load_model_gui],
        queue=True,
        show_progress="minimal",
    ).success(
        fn=sd_gen_generate_pipeline,  # fn=sd_gen.generate_pipeline,
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
            retain_task_cache_gui,
            image_ip1,
            mask_ip1,
            model_ip1,
            mode_ip1,
            scale_ip1,
            image_ip2,
            mask_ip2,
            model_ip2,
            mode_ip2,
            scale_ip2,
            pag_scale_gui,
            load_lora_cpu_gui,
            verbose_info_gui,
            gpu_duration_gui,
        ],
        outputs=[result_images, actual_task_info],
        queue=True,
        show_progress="minimal",
    )

app.queue()

app.launch(
    show_error=True,
    debug=True,
    allowed_paths=["./images/"],
)