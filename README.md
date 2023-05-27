# layered-depth-refinement

Official code for Layered Depth Refinement with Mask Guidance (CVPR 2022)
[[paper]](https://arxiv.org/abs/2206.03048) [[project page]](https://sooyekim.github.io/MaskDepth/)

If you find our repository useful, please consider citing our paper:
```
@inproceedings{kim2022layered,
  title     = {Layered Depth Refinement with Mask Guidance},
  author    = {Kim, Soo Ye and Zhang, Jianming and Niklaus, Simon and Fan, Yifei and Lin, Zhe and Kim, Munchurl},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022}
  }
```
## Prerequisites
### Install
* numpy
* cv2
* matplotlib
* timm
* torch
* torchvision
* mmcv-full
    * Can be installed by specifying a specific version (see section below), or by:  
    ```
    pip install -U openmim
    mim install mmcv-full
    ```

### Tested using
* CUDA 10.2
* Python 3.6.10
* PyTorch 1.9.1
* mmcv-full 1.3.14
```
pip install mmcv-full==1.3.14 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html
```
Other PyTorch versions should also work with compatible Python-PyTorch-CUDA combinations (for GPU) or Python-PyTorch combinations (for CPU).

### Prepare test images and masks
Some example test images are provided in `./images/input/rgb` with initial depth maps generated using DPT (Ranftl _et al_., Vision Transformers for Dense Prediction, ICCV 2021) in `./images/input/depth`. Corresponding masks can be generated by any preferred masking tool and added to `./images/input/mask`.

## Usage
Run `test.sh` to refine initial depth maps in `./images/input/depth` using masks in `./images/input/mask` and RGB images in `./images/input/rgb`. Results will be saved in `./images/output`.  
Different directories can be given as arguments with the following command:
```
python test.py \
--ckpt_path <ckpt_path> \
--test_input_rgb_dir <RGB input directory> \
--test_input_depth_dir <depth input directory> \
--test_input_mask_dir <mask input directory> \
--test_output_dir <output directory>
```

### Notes:
* Hard masks (binary) and soft masks (alpha matte) are both acceptable with ranges in [0, 255].
* High-resolution outputs can be saved with the flag `--hires`.
* `--input_size 320` often works better than 512.
* See other testing parameters in `test.py`.

## Results
![example3](https://github.com/adobe-research/layered-depth-refinement/assets/25021245/1d5ee445-a2e6-44e1-9d33-42b805413aee)

[1] Ranftl _et al_., Vision Transformers for Dense Prediction, ICCV, 2021.

More examples can be found on our [project page](https://sooyekim.github.io/MaskDepth/)!

## License
The `MiT_siamese.py` file is licensed under the [**NVIDIA Source Code License for SegFormer**](https://git.corp.adobe.com/sooyek/layered-depth-refinement/blob/main/LICENSE.md#nvidia-source-code-license-for-segformer). The `test.py` file and all other materials, including the model checkpoint and shell script, are licensed under the [**Adobe Research License**](https://git.corp.adobe.com/sooyek/layered-depth-refinement/blob/main/LICENSE.md#adobe-research-license). Images in `./images/input/rgb` (also shown in Results above) were licensed from www.unsplash.com under the [**standard Unsplash license**](https://unsplash.com/license).
