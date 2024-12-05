from PIL import Image
import math
import os
from pytorch_msssim import ssim, ms_ssim
import matplotlib.pyplot as plt

from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms
from diffusers import DDIMInverseScheduler
import torch

def show_images_side_by_side(images, titles=None, figsize=(8,4)):
    """
    Display a list of images side by side.
    
    Args:
    images (list of numpy arrays): List of images to display.
    titles (list of str, optional): List of titles for each image. Default is None.
    """
    num_images = len(images)
    
    if titles is not None:
        if len(titles) != num_images:
            raise ValueError("Number of titles must match the number of images.")
    
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i]) 
        ax.axis('off')
        
        if titles is not None:
            ax.set_title(titles[i])
    
    plt.tight_layout()
    plt.show()
    return

def show_latent_and_final_img(latent: torch.Tensor, img: torch.Tensor, pipe):
    with torch.no_grad():
        latents_pil_img = pipe.numpy_to_pil(pipe.decode_latents(latent.detach()))[0]
        pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    show_images_side_by_side([latents_pil_img, pil_img], ['Latent','Generated Image'])
    return

def save_img(path, img: torch.Tensor, pipe):
    pil_img = pipe.numpy_to_pil(pipe.img_tensor_to_numpy(img))[0]
    pil_img.save(path)
    return

def get_img_tensor(img_path, device, size=None):
    if size is None:
        img_tensor = pil_to_tensor(Image.open(img_path).convert("RGB"))/255
    else:
        img_tensor = pil_to_tensor(Image.open(img_path).resize(size).convert("RGB"))/255
    return img_tensor.unsqueeze(0).to(device)

def create_output_folder(cfgs):
    parent = os.path.join(cfgs['save_img'], cfgs['dataset'])
    wm_path = os.path.join(parent, cfgs['method'], cfgs['case'])
    
    special_model = ['CompVis']
    for key in special_model:
        if key in cfgs['model_id']:
            wm_path = os.path.join(parent, cfgs['method'], '_'.join([cfgs['case'][:-1], key+'/']))
            break
        
    os.makedirs(wm_path, exist_ok=True)
    ori_path = os.path.join(parent, 'OriImgs/')
    os.makedirs(ori_path, exist_ok=True)
    return wm_path, ori_path

# Metrics for similarity
def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return 100
    return 20 * math.log10(1.) - 10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_ssim(a, b):
    return ssim(a, b, data_range=1.).item()

def load_img(img_path, device):
    img = Image.open(img_path).convert('RGB')
    x = (transforms.ToTensor()(img)).unsqueeze(0).to(device)
    return x

def eval_psnr_ssim_msssim(ori_img_path, new_img_path, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return compute_psnr(ori_x, new_x), compute_ssim(ori_x, new_x), compute_msssim(ori_x, new_x)

def eval_lpips(ori_img_path, new_img_path, metric, device):
    ori_x, new_x = load_img(ori_img_path, device), load_img(new_img_path, device)
    return metric(ori_x, new_x).item()

# Detect watermark from one image
def watermark_prob(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cuda'), size=None):
    if isinstance(img, str):
        if size is None:
            img_tensor = pil_to_tensor(Image.open(img).convert("RGB"))/255
        else:
            img_tensor = pil_to_tensor(Image.open(img).resize(size).convert("RGB"))/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img

    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    reversed_latents = dect_pipe.forward_diffusion(
        latents=img_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1.0,
        num_inference_steps=50,
    )
    det_prob = wm_pipe.one_minus_p_value(reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    return det_prob

def watermark_prob_nodiffusion(img, dect_pipe, wm_pipe, text_embeddings, tree_ring=True, device=torch.device('cuda')):
    if isinstance(img, str):
        img_tensor = pil_to_tensor(Image.open(img).convert("RGB"))/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img

    img_latents = dect_pipe.get_image_latents(img_tensor, sample=False)
    det_prob = wm_pipe.one_minus_p_value(img_latents) if not tree_ring else wm_pipe.tree_ring_p_value(img_latents)
    return det_prob

## new helpers ##
@torch.no_grad()
def get_text_embedding(pipe, prompt):
    text_input_ids = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]
    return text_embeddings

# The reverse of decode_latents_tensor()
@torch.no_grad()
def get_image_latents(pipe, image: torch.Tensor, sample=True, rng_generator=None):
    image = 2.0 * image - 1.0
    encoding_dist = pipe.vae.encode(image).latent_dist
    if sample:
        encoding = encoding_dist.sample(generator=rng_generator)
    else:
        encoding = encoding_dist.mode()
    latents = encoding * pipe.vae.config.scaling_factor
    return latents

def get_init_latent(img_tensor, pipe, curr_scheduler):
    # DDIM inversion from the given image
    img_latents = get_image_latents(pipe, img_tensor, sample=False)
    pipe.scheduler = DDIMInverseScheduler.from_config(curr_scheduler.config)
    reversed_latents = pipe(
                            prompt='',
                            latents=img_latents,
                            guidance_scale=1,
                            num_inference_steps=50,
                            output_type='latent',
                        )
    reversed_latents = reversed_latents.images.float()
    pipe.scheduler = curr_scheduler
    return reversed_latents

def watermark_prob_new(img, pipe, wm_pipe, tree_ring=True, device=torch.device('cuda'), size=None):
    if isinstance(img, str):
        if size is None:
            img_tensor = pil_to_tensor(Image.open(img).convert("RGB"))/255
        else:
            img_tensor = pil_to_tensor(Image.open(img).resize(size).convert("RGB"))/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
    elif isinstance(img, torch.Tensor):
        img_tensor = img
    
    curr_scheduler = pipe.scheduler
    reversed_latents = get_init_latent(img_tensor, pipe, curr_scheduler)
    det_prob = wm_pipe.one_minus_p_value(reversed_latents) if not tree_ring else wm_pipe.tree_ring_p_value(reversed_latents)
    return det_prob