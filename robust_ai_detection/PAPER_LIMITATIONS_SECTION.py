"""
PAPER_LIMITATIONS_SECTION.md

Ready-to-use text for the Limitations section of your paper.
This honestly addresses the CPU/resource constraints while
framing them as proper academic limitations.
"""

LIMITATIONS_TEXT = """
## Limitations

### Computational Constraints
All experiments were conducted on a consumer-grade laptop equipped with 
an Intel Core i5-10th Generation processor and 16GB DDR4 RAM, without 
a dedicated GPU. This imposed practical limitations on the scale of 
experiments: we used a stratified subset of 2,000 images per class for 
training and evaluation, rather than the full dataset. The CLIP-based 
variant of our detector (as used in the base paper) employs ViT-B/32 
instead of ViT-L/14 to remain computationally feasible on CPU hardware. 
These constraints may cause our reported metrics to be slightly lower 
than those achievable with full dataset training on GPU-accelerated 
infrastructure.

### Cross-Generator Shift
Our cross-generator shift evaluation is limited to analyzing the model's 
behavior on Stable Diffusion-generated images (CIFAKE dataset). A more 
comprehensive study would include images from GANs (e.g., StyleGAN2, 
ProGAN), additional diffusion architectures (DALL-E, GLIDE, Midjourney), 
and autoregressive models. We leave broader cross-generator evaluation 
for future work with access to larger computational resources.

### Saliency Interpretation
We use Vanilla Gradient saliency maps as our primary explainability tool. 
While straightforward and computationally lightweight, vanilla gradients 
can be noisy compared to more sophisticated attribution methods such as 
Grad-CAM, Integrated Gradients, or SHAP. Future work should investigate 
whether more robust attribution methods reveal different failure patterns 
under distribution shifts.

### Dataset Scope
Experiments are conducted on CIFAKE, which contains 32×32 pixel images. 
Real-world forensics scenarios typically involve higher-resolution images. 
While the detection patterns we study (compression artifacts, generator 
fingerprints) are resolution-independent in principle, validation on 
high-resolution datasets (e.g., FFHQ at 1024×1024) is warranted.

### Reproducibility Note
All code, hyperparameters, and random seeds are fixed and documented 
(seed=42). The complete codebase is structured for full reproducibility. 
Results obtained on different hardware may vary slightly due to 
floating-point non-determinism in CPU execution, but the trends and 
relative comparisons across distribution shifts are expected to be robust.
"""

if __name__ == "__main__":
    print(LIMITATIONS_TEXT)
