# LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization <br><sub>Official PyTorch Implementation</sub>

#### [<code>HF Checkpoint 🚀</code>](https://huggingface.co/Beckham808/LightGen) | [<code>Technical Report 📝</code>](http://arxiv.org/abs/)  

  <p align="center">
    LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization
    <br />
    <a href="https://maradona10wxf.github.io/">Xianfeng Wu<sup>1, 2</sup><sup>#</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=0bmTpcAAAAAJ&hl=en&oi=ao">Yajing Bai<sup>1, 2</sup><sup>#</sup></a>
    ·
    <a href="https://sairlab.org/haozez/">Haoze Zheng<sup>1, 2</sup><sup>#</sup></a>
    ·
    <a href="https://haroldchen19.github.io/">Harold (haodong) Chen<sup>1, 2</sup><sup>#</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=Y8zBpcoAAAAJ&hl=zh-CN">Yexin Liu<sup>1, 2</sup><sup>#</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=UhFbFCMAAAAJ&hl=en">Zihao Wang<sup>1, 2</sup></a>
    ·
    <a href="">Xuran Ma<sup>1, 2</sup></a>
    ·
    <a href="https://scholar.google.cz/citations?user=bM_lvLAAAAAJ&hl=zh-CN">Wenjie Shu<sup>1, 2</sup></a>
    ·
    <a href="">Xianzu Wu<sup>1, 2</sup></a>
    ·
    <a href="https://leehomyc.github.io/">Harry Yang<sup>1, 2</sup><sup>*</sup></a>
    ·
    <a href="https://scholar.google.com/citations?user=HX0BfLYAAAAJ&hl=en">Sernam Lim<sup>2, 3</sup><sup>*</sup></a>
    <br />
    <p align="center"> <sub><sup>1</sup> <a href="https://amc.hkust.edu.hk/">HKUST AMC<sup></a>, <sup>2</sup> <a href="https://www.everlyn.ai/">Everlyn AI<sup></a>, <sup>3</sup> <a href="https://www.cs.ucf.edu/">UCF CS<sup></a>, <sup>#</sup>Equal contribution, <sup>*</sup> Corresponding Author</sub></p>
  </p>

<p align="center">
  <img src="demo/demo.png" width="720">
</p>

This is a PyTorch/GPU implementation of [LightGen](https://arxiv.org/pdf/):
this repo wants to provide a efficient pre-training pipeline for text-to-image generation on [Fluid](https://arxiv.org/pdf/2410.13863) and [MAR](https://github.com/LTH14/mar)

## 🦉 ToDo List

- [ ] DPO Post-proceesing Code Released 
- [ ] Release Complete Checkpoint.
- [ ] Add Accelerate Module.

## Env

```bash
conda create -n everlyn_video python=3.10
conda activate everlyn_video
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# pip install -U xformers==0.0.26 --index-url https://download.pytorch.org/whl/cu121
pip install -r requierments.txt
```

## Prepare stage

```bash
huggingface-cli download --token hf_ur_token --resume-download stabilityai/stable-diffusion-3.5-large --local-dir stable-diffusion-3.5-large # Image VAE
huggingface-cli download --resume-download google/flan-t5-xxl --local-dir google/flan-t5-xxl # Text Encoder
huggingface-cli download --repo-type dataset --resume-download jackyhate/text-to-image-2M --local-dir text-to-image-2M # Dataset
```
untar script for text-to-image2M
```bash
#!/bin/bash

# Check if the 'untar' directory exists, and create it if it does not
mkdir -p untar

# Loop through all .tar files
for tar_file in *.tar; do
    # Extract the numeric part, for example 00001, 00002, ...
    dir_name=$(basename "$tar_file" .tar)
    
    # Create the corresponding directory
    mkdir -p "untar/$dir_name"
    
    # Extract the tar file to the corresponding directory
    tar -xvf "$tar_file" -C "untar/$dir_name"
    
    echo "Extraction completed: $tar_file to untar/$dir_name"
done

echo "All files have been extracted."
```
It may too large to cost much time to read this data in normal dataset, so we need to generate a json file first 
to accelerate this process, modify `scripts/generate_txt.py` then run it.

```bash
python generate_json.py
```

## Training
Script for the default setting, u can modify some setting in `scripts/run.sh`:
```bash
sh run.sh
```
<!-- `diffusion/__init__.py` maybe need reduce the time step -->

## Inference
Script for the default setting:
```bash
python pipeline_image.py
```

## Acknowledgements

A large portion of codes in this repo is based on [MAR](https://github.com/LTH14/mar).
