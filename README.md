## Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement (ICLR 2023 Oral)

[Paper]( ) | [Project Page](https://li-chongyi.github.io/UHDFour/) 

[Chongyi Li](https://li-chongyi.github.io/), [Chun-Le Guo](https://scholar.google.com.au/citations?user=RZLYwR0AAAAJ&hl=en),  [Man Zhou](https://manman1995.github.io/),  [Zhexin Liang](https://zhexinliang.github.io/),  [Shangchen Zhou](https://shangchenzhou.com/),  [Ruicheng Feng](https://jnjaby.github.io/),   [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

S-Lab, Nanyang Technological University; Nankai University

### Updates

- **2023.01.30**:  This repo is created.


---

### UHD-LL Dataset
(The datasets are hosted on both Google Drive and BaiduPan)
| Dataset | Link | Number | Description|
| :----- | :--: | :----: | :---- | 
| UHD-LL| [Google Drive]() / [BaiduPan (key: dz6u)]() | 2,150 | A total of 2,000 pairs for training and 150 pairs for testing.|


<details close>
<summary>[Unfold] for detailed description of each folder in UHD-LL dataset:</summary>

<table>
<td>

| UDH-LL               | Description             |
| :----------------------- | :---------------------- |
| training_set/gt                 | normal-light images |
| training_set/input          | low-light  images |
| testing_set/gt               | normal-light images |
| testing_set/input          |low-light  images |

</td>
</table>


</details>


### Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/sczhou/LEDNet
cd LEDNet

# create new anaconda env
conda create -n lednet python=3.8 -y
conda activate lednet

# install python dependencies
pip3 install -r requirements.txt
python basicsr/setup.py develop
```


### Train the Model
Before training, you need to:

- Download the LOL-Blur Dataset from [Google Drive](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX?usp=sharing) / [BaiduPan (key: dz6u)](https://pan.baidu.com/s/1CPphxCKQJa_iJAGD6YACuA).
- Specify `dataroot_gt` and `dataroot_lq` in the corresponding option file.

Training LEDNet:
```
# without GAN
python basicsr/train.py -opt options/train_LEDNet.yml

# with GAN
python basicsr/train.py -opt options/train_LEDNetGAN.yml
```
This project is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), the detailed tutorial on training commands and config settings can be found [here](https://github.com/XPixelGroup/BasicSR/blob/master/docs/introduction.md).

### Quick Inference
- Download the LEDNet pretrained model from [[Release V0.1.0](https://github.com/sczhou/LEDNet/releases/tag/v0.1.0)] to the `weights` folder. You can manually download the pretrained models OR download by runing the following command.
  
  > python scripts/download_pretrained_models.py LEDNet
  
Inference LEDNet:
```
# test LEDNet (paper model)
python inference_lednet.py --model lednet --test_path ./inputs

# test retrained LEDNet (higher PSNR and SSIM)
python inference_lednet.py --model lednet_retrain --test_path ./inputs

# test LEDNetGAN
python inference_lednet.py --model lednetgan --test_path ./inputs
```
The results will be saved in the `results` folder.

### Evaluation

```
# set evaluation metrics of 'psnr', 'ssim', and 'lpips (vgg)'
python scripts/calculate_iqa_pair.py --result_path 'RESULT_ROOT' --gt_path 'GT_ROOT' --metrics psnr ssim lpips
```
(The released model was retrained using the [BasicSR](https://github.com/XPixelGroup/BasicSR) framework, which makes it easier to use or further develop upon this work. NOTE that the PSNR and SSIM scores of retrained model are higher than the paper model.)

### Generate Low-light Images from Your Own Data
- Download the CE-ZeroDCE pretrained model from [[Release V0.1.0](https://github.com/sczhou/LEDNet/releases/tag/v0.1.0)] to the `weights` folder. You can manually download the pretrained models OR download by runing the following command.
  
  > python scripts/download_pretrained_models.py CE-ZeroDCE
  
Run low-light generation:
```
python scripts/generate_low_light_imgs.py --test_path 'IMG_ROOT' --result_path 'RESULT_ROOT' --model_path './weights/ce_zerodce.pth'
```

### Inference with Cog
To run containerized local inference with LEDNet using [Cog](https://github.com/replicate/cog), run the following commands in the project root:

```
cog run python basicsr/setup.py develop
cog predict -i image=@'path/to/input_image.jpg'
```

You can view this demo running as an API [here on Replicate](https://replicate.com/sczhou/lednet).

### License

This project is licensed under <a rel="license" href="https://github.com/sczhou/LEDNet/blob/master/LICENSE">S-Lab License 1.0</a>. Redistribution and use for non-commercial purposes should follow this license.

### Acknowledgement

This project is based on [BasicSR](https://github.com/XPixelGroup/BasicSR). We calculate evaluation metrics using [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) toolbox. Thanks for their awesome works.

### Citation
If our work is useful for your research, please consider citing:

```bibtex
@InProceedings{zhou2022lednet,
    author = {Zhou, Shangchen and Li, Chongyi and Loy, Chen Change},
    title = {LEDNet: Joint Low-light Enhancement and Deblurring in the Dark},
    booktitle = {ECCV},
    year = {2022}
}
```

### Contact
If you have any questions, please feel free to reach me out at `shangchenzhou@gmail.com`.
