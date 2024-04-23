# GANGenerationDetection_F23

Authors: Avi, Abhiram, Ashwin, Chris, and Kris

This repository is organized according to the number of layers in each GAN implementation. Thus, there are folders containing GAN implementations for 2, 5, 6, 8, and 10 layers. For more advanced architectures (such as TransGAN and Vision Transformer), there are separate folders for those.

Folders and their contents:

2Layer
* 2Layer4FilterGAN-Numbers.py
    - 2 layer GAN Implementation using MNIST handwritten numerical dataset.
* 2Layer128FilterGAN-Numbers.py
    - 2 layer GAN Implementation using MNIST handwritten numerical dataset.

5Layer
* 5Layer-64x64.py
* 5Layer-64x64x-Optimized.py

6Layer
* 6Layer-64x64.py
* 6Layer-128x128-Model-Loader.py
* 6Layer-128x128-Optimized.py
* 6Layer-128x128-Reconstruction.py
* 6Layer-128x128-SAGAN.py
* 6Layer-128x128-VEEGAN.py
* 6Layer-128x128-VEEGAN-2.py
* 6Layer-128x128-VQGAN.py
* 6Layer-128x128-Wasserstein.py
* 6Layer-128x128.py
* MoreTrainSAGAN.py
* ProduceImages.py

8Layer
* 8Layer-128x128-Feature-Matching.py
* 8Layer-128x128-Minibatch-Discrimination.py
* 8Layer-128x128-Reconstruction.py
* 8Layer-128x128-SAGAN.py
* 8Layer-128x128-Unrolled.py
* 8Layer-128x128.py

10Layer
* 10Layer-128x128-SAGAN.py

TransGAN
* CelebADataset.py
* Discriminator.py
* Generator.py
* GridTransformerBlock.py
* Helpers.py
* PositionalEncoding.py

* Train.py
* TransformerBlock.py

ViT
* cifar10.py
