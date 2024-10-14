



# Train SAM 2：训练/微调 Segment Anything 2 (指南/代码)

该仓库包含微调/训练 Segment Anything 2 的代码。

训练脚本可以在TRAIN.py 中找到.请注意，TRAIN.py 每个批次使用单张图像，而TRAIN_multi_image_batch_fast.py每个批次使用多张图像，除此之外它们是相同的。

加载和使用微调网络的代码可以在TEST_Net_linux.py 中找到。

processed_mask_v2.py用来处理合并mask

My_Test.py用来测试微调模型效果

训练是在 [LabPics 1 数据集](https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1) 上完成的。



# Original readme

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

![SA-V dataset](assets/sa_v_dataset.jpg?raw=true)

## Installation

Please install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git

cd segment-anything-2; pip install -e .
```

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[demo]"
```

## Getting Started

### Download Checkpoints

- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)

  

## Model Description

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny    |     38.9     |          47.2          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small   |      46      | 43.3 (53.0 compiled\*) |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus |     80.8     | 34.8 (43.8 compiled\*) |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large   |    224.4     | 24.2 (30.2 compiled\*) |        76.0         |        74.6        |       79.8        |

\* Compile the model by setting `compile_image_encoder: True` in the config.

## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

