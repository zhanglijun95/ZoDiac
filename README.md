# Attack-Resilient Image Watermarking Using Stable Diffusion (NeurIPS2024)

This is the website for our paper "Attack-Resilient Image Watermarking Using Stable Diffusion". 
The arXiv version can be found [here](https://arxiv.org/pdf/2401.04247.pdf).

### Abstract
Watermarking images is critical for tracking image provenance and proving ownership. With the advent of generative models, such as stable diffusion, that can create fake but realistic images, watermarking has become particularly important to make human-created images reliably identifiable. Unfortunately, the very same stable diffusion technology can remove watermarks injected using existing methods. To address this problem, we present ZoDiac, which uses a pre-trained stable diffusion model to inject a watermark into the trainable latent space, resulting in watermarks that can be reliably detected in the latent vector even when attacked. We evaluate ZoDiac on three benchmarks, MS-COCO, DiffusionDB, and WikiArt, and find that ZoDiac is robust against state-of-the-art watermark attacks, with a watermark detection rate above 98% and a false positive rate below 6.4%, outperforming state-of-the-art watermarking methods. We hypothesize that the reciprocating denoising process in diffusion models may inherently enhance the robustness of the watermark when faced with strong attacks and validate the hypothesis. Our research demonstrates that stable diffusion is a promising approach to robust watermarking, able to withstand even stable-diffusion–based attack methods. 


### Cite
Welcome to cite our work if you find it is helpful to your research.
```
@misc{zhang2024ZoDiac,
      title={Attack-Resilient Image Watermarking Using Stable Diffusion}, 
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
