# Robust Image Watermarking using Stable Diffusion
This is the website for our paper "Robust Image Watermarking using Stable Diffusion". 
The arXiv version can be found [here](https://arxiv.org/pdf/2401.04247.pdf).

### Abstract
Watermarking images is critical for tracking image provenance and claiming ownership. With the advent of generative models, such as stable diffusion, able to create fake but realistic images, watermarking has become particularly important, e.g., to make generated images reliably identifiable. Unfortunately, the very same stable diffusion technology can remove watermarks injected using existing methods. To address this problem, we present a ZoDiac, which uses a pre-trained stable diffusion model to inject a watermark into the trainable latent space, resulting in watermarks that can be reliably detected in the latent vector, even when attacked. We evaluate ZoDiac on three benchmarks, MS-COCO, DiffusionDB, and WikiArt, and find that ZoDiac is robust against state-of-the-art watermark attacks, with a watermark detection rate over 98% and a false positive rate below 6.4%, outperforming state-of-theart watermarking methods. Our research demonstrates that stable diffusion is a promising approach to robust watermarking, able to withstand even stable-diffusion-based attacks.


### Cite
Welcome to cite our work if you find it is helpful to your research.
```
@misc{zhang2024robust,
      title={Robust Image Watermarking using Stable Diffusion}, 
      author={Lijun Zhang and Xiao Liu and Antoni Viros Martin and Cindy Xiong Bearfield and Yuriy Brun and Hui Guan},
      year={2024},
      eprint={2401.04247},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# How To Use
Prepare the conda environment by running ```conda env create -f environment.yml```.
Then please refer to the ```Example.ipynb```. Each section can be executed seperately.