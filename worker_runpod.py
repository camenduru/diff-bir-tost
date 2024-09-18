import os, shutil, json, requests, random, runpod

import torch
from accelerate.utils import set_seed
from utils.inference import V1InferenceLoop, BSRInferenceLoop, BFRInferenceLoop, UnAlignedBFRInferenceLoop, BIDInferenceLoop

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Args(
    task=None,
    upscale=None,
    version="v2",
    steps=50,
    better_start=False,
    tiled=False,
    tile_size=512,
    tile_stride=256,
    pos_prompt="",
    neg_prompt="low quality, blurry, low-resolution, noisy, unsharp, weird textures",
    cfg_scale=4.0,
    input=None,
    n_samples=1,
    guidance=False,
    g_loss="w_mse",
    g_scale=0.0,
    g_start=1001,
    g_stop=-1,
    g_space="latent",
    g_repeat=1,
    output=None,
    seed=231,
    device="cuda"
)

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    original_file_name = url.split('/')[-1]
    _, original_file_extension = os.path.splitext(original_file_name)
    file_path = os.path.join(save_dir, file_name + original_file_extension)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image_check']
    input_image = download_file(url=input_image, save_dir='/content/input', file_name='diffbir_tost')

    args.input=input_image
    args.task=values['task']
    args.upscale=values['upscale']
    args.version=values['version']
    args.steps=values['steps']
    args.better_start=values['better_start']
    args.tiled=values['tiled']
    args.tile_size=values['tile_size']
    args.tile_stride=values['tile_stride']
    args.pos_prompt=values['pos_prompt']
    args.neg_prompt=values['neg_prompt']
    args.cfg_scale=values['cfg_scale']
    args.guidance=values['guidance']
    args.g_loss=values['g_loss']
    args.g_scale=values['g_scale']
    args.g_space=values['g_space']
    args.seed=values['seed']
    args.output='/content/result'
    
    set_seed(args.seed)
    if args.version == "v1":
        V1InferenceLoop(args).run()
    else:
        supported_tasks = {
            "sr": BSRInferenceLoop,
            "dn": BIDInferenceLoop,
            "fr": BFRInferenceLoop,
            "fr_bg": UnAlignedBFRInferenceLoop
        }
        supported_tasks[args.task](args).run()

    if args.task == "fr_bg":
        result = "/content/result/restored_backgrounds/diffbir_tost.png"
    else:
        result = "/content/result/diffbir_tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists('/content/input'):
            shutil.rmtree('/content/input')
        if os.path.exists('/content/result'):
            shutil.rmtree('/content/result')

runpod.serverless.start({"handler": generate})