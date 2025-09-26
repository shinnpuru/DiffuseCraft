import os
from stablepy.diffusers_vanilla.constants import FLUX_CN_UNION_MODES
from stablepy import (
    scheduler_names,
    SD15_TASKS,
    SDXL_TASKS,
    ALL_BUILTIN_UPSCALERS,
    IP_ADAPTERS_SD,
    IP_ADAPTERS_SDXL,
)

# - **Download Models**
DOWNLOAD_MODEL = ""

# - **Download VAEs**
DOWNLOAD_VAE = ""

# - **Download LoRAs**
DOWNLOAD_LORA = ""

LOAD_DIFFUSERS_FORMAT_MODEL = [
    'stabilityai/stable-diffusion-xl-base-1.0',
    'black-forest-labs/FLUX.1-dev',
    'black-forest-labs/FLUX.1-schnell',
]

DIFFUSERS_FORMAT_LORAS = [
]

DOWNLOAD_EMBEDS = [
]

CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY")
HF_TOKEN = os.environ.get("HF_READ_TOKEN")

DIRECTORY_MODELS = 'models'
DIRECTORY_LORAS = 'loras'
DIRECTORY_VAES = 'vaes'
DIRECTORY_EMBEDS = 'embedings'
DIRECTORY_UPSCALERS = 'upscalers'

CACHE_HF = "huggingface/hub/"
STORAGE_ROOT = "."

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
    'repaint ControlNet': 'repaint',
}

TASK_MODEL_LIST = list(TASK_STABLEPY.keys())

UPSCALER_DICT_GUI = {
    None: None,
    **{bu: bu for bu in ALL_BUILTIN_UPSCALERS if bu not in ["HAT x4", "DAT x4", "DAT x3", "DAT x2", "SwinIR 4x"]},
    # "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    # "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    # "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    # "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    # "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    # "realesr-general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "4x-UltraSharp": "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    "4x_foolhardy_Remacri": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    "Remacri4xExtraSmoother": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    "AnimeSharp4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    "lollypop": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    "RealisticRescaler4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    "NickelbackFS4x": "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth"
}

UPSCALER_KEYS = list(UPSCALER_DICT_GUI.keys())

DIFFUSERS_CONTROLNET_MODEL = [
    "Automatic",

    # "brad-twinkl/controlnet-union-sdxl-1.0-promax",
    # "xinsir/controlnet-union-sdxl-1.0",
    # "xinsir/anime-painter",
    # "Eugeoter/noob-sdxl-controlnet-canny",
    # "Eugeoter/noob-sdxl-controlnet-lineart_anime",
    # "Eugeoter/noob-sdxl-controlnet-depth",
    # "Eugeoter/noob-sdxl-controlnet-normal",
    # "Eugeoter/noob-sdxl-controlnet-softedge_hed",
    # "Eugeoter/noob-sdxl-controlnet-scribble_pidinet",
    # "Eugeoter/noob-sdxl-controlnet-scribble_hed",
    # "Eugeoter/noob-sdxl-controlnet-manga_line",
    # "Eugeoter/noob-sdxl-controlnet-lineart_realistic",
    # "Eugeoter/noob-sdxl-controlnet-depth_midas-v1-1",
    # "dimitribarbot/controlnet-openpose-sdxl-1.0-safetensors",
    "r3gm/controlnet-openpose-sdxl-1.0-fp16",
    "r3gm/controlnet-canny-scribble-integrated-sdxl-v2-fp16",
    "r3gm/controlnet-union-sdxl-1.0-fp16",
    "r3gm/controlnet-lineart-anime-sdxl-fp16",
    "r3gm/control_v1p_sdxl_qrcode_monster_fp16",
    "r3gm/controlnet-tile-sdxl-1.0-fp16",
    "r3gm/controlnet-recolor-sdxl-fp16",
    "r3gm/controlnet-openpose-twins-sdxl-1.0-fp16",
    "r3gm/controlnet-qr-pattern-sdxl-fp16",
    # "Yakonrus/SDXL_Controlnet_Tile_Realistic_v2",
    # "TheMistoAI/MistoLine",
    # "briaai/BRIA-2.3-ControlNet-Recoloring",
    # "briaai/BRIA-2.3-ControlNet-Canny",

    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11p_sd15_mlsd",
    "lllyasviel/control_v11p_sd15_scribble",
    "lllyasviel/control_v11p_sd15_softedge",
    "lllyasviel/control_v11p_sd15_seg",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "lllyasviel/control_v11e_sd15_shuffle",
    "lllyasviel/control_v11e_sd15_ip2p",
    "lllyasviel/control_v11p_sd15_inpaint",
    # "monster-labs/control_v1p_sd15_qrcode_monster",
    # "lllyasviel/control_v11f1e_sd15_tile",
    # "latentcat/control_v1p_sd15_brightness",
    # "yuanqiuye/qrcode_controlnet_v3",

    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    # "Shakker-Labs/FLUX.1-dev-ControlNet-Pose",
    # "Shakker-Labs/FLUX.1-dev-ControlNet-Depth",
    # "jasperai/Flux.1-dev-Controlnet-Upscaler",
    # "jasperai/Flux.1-dev-Controlnet-Depth",
    # "jasperai/Flux.1-dev-Controlnet-Surface-Normals",
    # "XLabs-AI/flux-controlnet-canny-diffusers",
    # "XLabs-AI/flux-controlnet-hed-diffusers",
    # "XLabs-AI/flux-controlnet-depth-diffusers",
    # "InstantX/FLUX.1-dev-Controlnet-Union",
    # "InstantX/FLUX.1-dev-Controlnet-Canny",
]

PROMPT_W_OPTIONS = [
    ("Compel format: (word)weight", "Compel"),
    ("Classic format: (word:weight)", "Classic"),
    ("Classic-original format: (word:weight)", "Classic-original"),
    ("Classic-no_norm format: (word:weight)", "Classic-no_norm"),
    ("Classic-sd_embed format: (word:weight)", "Classic-sd_embed"),
    ("Classic-ignore", "Classic-ignore"),
    ("None", "None"),
]

WARNING_MSG_VAE = (
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

DIFFUSECRAFT_CHECKPOINT_NAME = {
    "sd1.5": "SD 1.5",
    "sdxl": "SDXL",
    "flux-dev": "FLUX",
    "flux-schnell": "FLUX",
}

POST_PROCESSING_SAMPLER = ["Use same sampler"] + [
    name_s for name_s in scheduler_names if "Auto-Loader" not in name_s
]

IP_MODELS = []
ALL_IPA = sorted(set(IP_ADAPTERS_SD + IP_ADAPTERS_SDXL))

for origin_name in ALL_IPA:
    suffixes = []
    if origin_name in IP_ADAPTERS_SD:
        suffixes.append("sd1.5")
    if origin_name in IP_ADAPTERS_SDXL:
        suffixes.append("sdxl")
    ref_name = f"{origin_name} ({'/'.join(suffixes)})"
    IP_MODELS.append((ref_name, origin_name))

MODE_IP_OPTIONS = ["original", "style", "layout", "style+layout"]

SUBTITLE_GUI = (
    "### This demo uses [diffusers](https://github.com/huggingface/diffusers)"
    " to perform different tasks in image generation."
)

HELP_GUI = (
    """### Help:
    - The current space runs on a ZERO GPU which is assigned for approximately 60 seconds; Therefore, if you submit expensive tasks, the operation may be canceled upon reaching the maximum allowed time with 'GPU TASK ABORTED'.
    - Distorted or strange images often result from high prompt weights, so it's best to use low weights and scales, and consider using Classic variants like 'Classic-original'.
    - For better results with Pony Diffusion, try using sampler DPM++ 1s or DPM2 with Compel or Classic prompt weights.
    """
)

EXAMPLES_GUI_HELP = (
    """### The following examples perform specific tasks:
    1. Generation with SDXL and upscale
    2. Generation with FLUX dev
    3. ControlNet Canny SDXL
    4. Optical pattern (Optical illusion) SDXL
    5. Convert an image to a coloring drawing
    6. V prediction model inference
    7. V prediction model sd_embed variant inference
    8. ControlNet OpenPose SD 1.5 and Latent upscale

    - Different tasks can be performed, such as img2img or using the IP adapter, to preserve a person's appearance or a specific style based on an image.
    """
)

EXAMPLES_GUI = [
    [
        "splatter paint theme, 1girl, frame center, pretty face, face with artistic paint artwork, feminism, long hair, upper body view, futuristic expression illustrative painted background, origami, stripes, explosive paint splashes behind her, hand on cheek pose, strobe lighting, masterpiece photography creative artwork, golden morning light, highly detailed, masterpiece, best quality, very aesthetic, absurdres",
        "logo, artist name, (worst quality, normal quality), bad-artist, ((bad anatomy)), ((bad hands)), ((bad proportions)), ((duplicate limbs)), ((fused limbs)), ((interlocking fingers)), ((poorly drawn face)), high contrast., score_6, score_5, score_4, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
        28,
        5.0,
        -1,
        "None",
        0.33,
        "DPM++ 2M SDE",
        1152,
        896,
        "John6666/noobai-xl-nai-xl-epsilonpred10version-sdxl",
        "txt2img",
        "image.webp",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic-no_norm",
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
        "FlowMatch Euler",
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
        "DPM++ 2M SDE Ef",
        1024,
        1024,
        "John6666/epicrealism-xl-v10kiss2-sdxl",
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
        "SG161222/RealVisXL_V5.0",
        "optical pattern ControlNet",
        "spiral_no_transparent.png",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.05,  # cn start
        0.8,  # cn end
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
        "DPM++ 2M SDE",
        1024,
        1024,
        "eienmojiki/Anything-XL",
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
        "[mochizuki_shiina], [syuri22], newest, reimu, solo, outdoors, water, flower, lantern",
        "worst quality, normal quality, old, sketch,",
        28,
        7.0,
        -1,
        "None",
        0.33,
        "DPM 3M Ef",
        1600,
        1024,
        "Laxhar/noobai-XL-Vpred-1.0",
        "txt2img",
        "color_image.png",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic",
        None,
        30,
        False,
    ],
    [
        "[mochizuki_shiina], [syuri22], newest, multiple girls, 2girls, earrings, jewelry, gloves, purple eyes, black hair, looking at viewer, nail polish, hat, smile, open mouth, fingerless gloves, sleeveless, :d, upper body, blue eyes, closed mouth, black gloves, hands up, long hair, shirt, bare shoulders, white headwear, blush, black headwear, blue nails, upper teeth only, short hair, white gloves, white shirt, teeth, rabbit hat, star earrings, purple nails, pink hair, detached sleeves, fingernails, fake animal ears, animal hat, sleeves past wrists, black shirt, medium hair, fur trim, sleeveless shirt, turtleneck, long sleeves, rabbit ears, star \\(symbol\\)",
        "worst quality, normal quality, old, sketch,",
        28,
        7.0,
        -1,
        "None",
        0.33,
        "DPM 3M Ef",
        1600,
        1024,
        "Laxhar/noobai-XL-Vpred-1.0",
        "txt2img",
        "color_image.png",  # img conttol
        1024,  # img resolution
        0.35,  # strength
        1.0,  # cn scale
        0.0,  # cn start
        1.0,  # cn end
        "Classic-sd_embed",
        None,
        30,
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
        "DPM++ 2M SDE",
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
        "Classic-original",
        "Latent (antialiased)",
        46,
        False,
    ],
]

RESOURCES = (
    """### Resources
    - John6666's space has some great features you might find helpful [link](https://huggingface.co/spaces/John6666/DiffuseCraftMod).
    - You can also try the image generator in Colab’s free tier, which provides free GPU [link](https://github.com/R3gm/SD_diffusers_interactive).
    """
)