import os
import re
import gradio as gr
from constants import (
    DIFFUSERS_FORMAT_LORAS,
    CIVITAI_API_KEY,
    HF_TOKEN,
    MODEL_TYPE_CLASS,
    DIRECTORY_LORAS,
)
from huggingface_hub import HfApi
from diffusers import DiffusionPipeline
from huggingface_hub import model_info as model_info_data
from diffusers.pipelines.pipeline_loading_utils import variant_compatible_siblings
from pathlib import PosixPath


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
            download_things(DIRECTORY_LORAS, url, HF_TOKEN, CIVITAI_API_KEY)
    new_lora_model_list = get_model_list(DIRECTORY_LORAS)
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


def restart_space(repo_id: str, factory_reboot: bool, token: str):
    api = HfApi(token=token)
    api.restart_space(repo_id=repo_id, factory_reboot=factory_reboot)


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


def download_diffuser_repo(repo_name: str, model_type: str, revision: str = "main", token=True):

    variant = None
    if token is True and not os.environ.get("HF_TOKEN"):
        token = None

    if model_type == "SDXL":
        info = model_info_data(
            repo_name,
            token=token,
            revision=revision,
            timeout=5.0,
        )

        filenames = {sibling.rfilename for sibling in info.siblings}
        model_filenames, variant_filenames = variant_compatible_siblings(
            filenames, variant="fp16"
        )

        if len(variant_filenames):
            variant = "fp16"

    cached_folder = DiffusionPipeline.download(
        pretrained_model_name=repo_name,
        force_download=False,
        token=token,
        revision=revision,
        # mirror="https://hf-mirror.com",
        variant=variant,
        use_safetensors=True,
        trust_remote_code=False,
        timeout=5.0,
    )

    if isinstance(cached_folder, PosixPath):
        cached_folder = cached_folder.as_posix()

    # Task model
    # from huggingface_hub import hf_hub_download
    # hf_hub_download(
    #     task_model,
    #     filename="diffusion_pytorch_model.safetensors",  # fix fp16 variant
    # )

    return cached_folder


def progress_step_bar(step, total):
    # Calculate the percentage for the progress bar width
    percentage = min(100, ((step / total) * 100))

    return f"""
        <div style="position: relative; width: 100%; background-color: gray; border-radius: 5px; overflow: hidden;">
            <div style="width: {percentage}%; height: 17px; background-color: #800080; transition: width 0.5s;"></div>
            <div style="position: absolute; width: 100%; text-align: center; color: white; top: 0; line-height: 19px; font-size: 13px;">
                {int(percentage)}%
            </div>
        </div>
        """


def html_template_message(msg):
    return f"""
        <div style="position: relative; width: 100%; background-color: gray; border-radius: 5px; overflow: hidden;">
            <div style="width: 0%; height: 17px; background-color: #800080; transition: width 0.5s;"></div>
            <div style="position: absolute; width: 100%; text-align: center; color: white; top: 0; line-height: 19px; font-size: 14px; font-weight: bold; text-shadow: 1px 1px 2px black;">
                {msg}
            </div>
        </div>
        """
