# ActiveSGM: Semantics-driven Active Mapping

This is the Python implementation of the ActiveSGM with SplaTAM backbone. (**Understanding while Exploring:
Semantics-driven Active Mapping**. Published at Neurips 2025) [[Paper](https://arxiv.org/abs/2506.00225)]

## Environment

### Installation

We provide scripts to create the conda environment, and recommend running ActiveSGM with Python 3.8 and CUDA 11.7 or CUDA 12.1. Please modify the scripts as needed to match your GPU and CUDA version.

```bash
# Download
git clone --recursive https://github.com/lly00412/ActiveSGM

# Build conda environment (installs everything including tiny-cuda-nn and diff-gaussian-rasterization)
cd ActiveSGM
bash scripts/installation/conda_env/build_sem.sh
```

**Note:** The `build_sem.sh` script installs a complete environment including:
- PyTorch 1.13.1 with CUDA 11.7
- pytorch3d, habitat-sim
- CUDA C++ Standard Library (CCCL)
- tiny-cuda-nn
- diff-gaussian-rasterization-w-depth
- All other dependencies

### Build cuda tool for semantic rendering

#### dense-channel-rasterization
```bash
# clone from github (if not already cloned with --recursive)
git clone -b liyan/dev --single-branch https://github.com/lly00412/semantic-gaussians.git third_parties/channel_rasterization

# go to the submodule directory
cd ./third_parties/channel_rasterization/channel-rasterization/cuda_rasterizer

# modify config.h based on number of classes
# Edit the file and change: NUM_CHANNELS {num of class} // Default 3

# install the cuda tool
cd ../..
python setup.py install
pip install .
```

#### sparse-channel-rasterization
```bash
# clone from github (if not already cloned with --recursive)
git clone -b hairong/sparse_ver --single-branch https://github.com/lly00412/semantic-gaussians.git third_parties/sparse_channel_rasterization

# go to the submodule directory
cd ./third_parties/sparse_channel_rasterization/sparse-channel-rasterization/cuda_rasterizer

# modify config.h based on number of classes and logits to keep
# Edit the file and change:
# NUM_CHANNELS {num of class} // Default Replica: 102, MP3D: 41
# TOP_K_LOGITS_CHANNELS {number of logits to keep} // Default 16  

# install the cuda tool
cd ../..
python setup.py install
pip install .
```

## Data Preparation

### Dataset download
We run the experiments on [Replica](https://github.com/facebookresearch/Replica-Dataset/tree/main) and [Matterport3D](https://niessner.github.io/Matterport/)(MP3D) dataset using Habitat simulator, please follow the instruction of [ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER) to download these two datasets.

### Dataset configuration
After downloading the datasets, you need to configure the paths in the configuration files:

1. Update `configs/Replica/office0/habitat.py`:
   - Set `scene_id` to point to your Replica dataset location (e.g., `/mnt/data/replica/office_0/habitat/mesh_semantic.ply`)

2. Create Habitat scene configuration files:
   - Add `render_asset` and `collision_asset` to `{dataset_path}/office_0/habitat/replicaSDK_stage.stage_config.json`
   - Create `{dataset_path}/replica.scene_dataset_config.json` for Habitat scene dataset configuration

3. Create necessary symlinks (if needed):
   ```bash
   ln -s /path/to/your/dataset/office_0 data/Replica/office0
   touch data/Replica/office0/traj.txt
   ```

### Semantic mesh filtering for Matterport3D
We use Chamfer distance to remove floaters and generate clean semantic ground-truth meshes for MP3D scenes. Please run the following code before evaluation, and update the mesh file paths accordingly before running.

```bash
python src/data/filter_mesh_mp3d.py
```

### Generate finetuning data for OneFormer

We provide fine-tuned OneFormer checkpoints for [Replica](https://huggingface.co/lly00412/oneformer-replica-finetune) and [MP3D](https://huggingface.co/lly00412/oneformer-mp3d-finetune). If you would like to run ActiveSGM on your own data, we also include configuration files and scripts for generating finetuning data.

We use [generate_finetune_data.py](https://github.com/lly00412/ActiveSGM/blob/main/configs/Replica/generate_finetune_data.py) as the configuration to generate semantic observation via Habitat simulator.
To finetune OneFormer, please run the following script:
```
# Modify the custom data folder before running
bash scripts/finetune_mp3d_oneformer.sh
```

## Training

We train ActiveSGM on two NVIDIA RTX A6000 GPUs. 
GPU 0 ("device") is used for keyframe mapping and path planning, 
while GPU 1 ("semantic_device") handles the OneFormer interface and semantic rendering. 
You can modify the "device" and "semantic_device" fields in the configuration files to assign these tasks to different GPUs as needed.

**Note:** For single-GPU setup, use `0` instead of `0,1` and set both `device` and `semantic_device` to `cuda:0` in the config file.

```bash
# Run ActiveSGM on Replica
bash scripts/activesgm/run_replica.sh {SCENE} {NUM_RUN} {EXP} {ENABLE_VIS} {GPU_ID}

# Run ActiveSGM on Replica office0 (dual-GPU)
bash scripts/activesgm/run_replica.sh office0 1 ActiveSem 0 0,1

# Run ActiveSGM on Replica office0 (single-GPU)
bash scripts/activesgm/run_replica.sh office0 1 ActiveSem 0 0

# Run Splatam
bash scripts/activesgm/run_replica.sh office0 1 predefine 0 0,1

# Run SGS-SLAM
bash scripts/activesgm/run_replica.sh office0 1 sgsslam 0 0,1
```

## Evaluation

We evaluate ActiveSGM for 3D reconstruction, Semantic Segmentation and Novel View Synthesis.
```bash
# Evaluate 3D reconstruction
bash scripts/evaluation/eval_replica_3d.sh office0 1 ActiveSem 0 0,1

# Evaluate semantic segmentation
bash scripts/evaluation/eval_replica_semantic.sh office0 1 ActiveSem 0 0 0 final

# Evaluate novel view synthesis
bash scripts/evaluation/eval_replica_nvs_result.sh office0 1 ActiveSem 0 0,1
```

## Troubleshooting

### Common Issues

1. **CUDA C++ Standard Library not found** when building tiny-cuda-nn:
   - Make sure you ran `build_sem.sh` which installs CUDA CCCL package
   - Verify that `$CONDA_PREFIX/include/cuda/std` exists

2. **Empty reconstruction (0 Gaussian points)**:
   - Check that Habitat simulator can load the scene correctly
   - Verify `scene_dataset_config.json` and `stage_config.json` are properly configured

3. **NaN metrics (PSNR, Depth RMSE)**:
   - This is expected in active mode where ground truth frames are not available
   - Accuracy, Completeness, and Coverage metrics should have valid values

4. **GPU memory issues**:
   - Reduce the number of Gaussians by adjusting densification parameters
   - Use a single GPU setup if dual-GPU causes issues

For detailed setup instructions and known issues, see `SETUP_AND_USAGE.md`.

## Citation

```
@inproceedings{chen2025understanding,
  title={Understanding while Exploring: Semantics-driven Active Mapping},
  author={Chen, Liyan and Zhan, Huangying and Yin, Hairong and Xu, Yi and Mordohai, Philippos},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgement
We sincerely thank the owners of the following open source projects, which are used by our released codes:
[HabitatSim](https://github.com/facebookresearch/habitat-sim), 
[ActiveGAMER](https://github.com/oppo-us-research/ActiveGAMER), 
[OneFormer](https://github.com/SHI-Labs/OneFormer),
[SplaTAM](https://github.com/spla-tam/SplaTAM),
[Semantic Gaussians](https://github.com/sharinka0715/semantic-gaussians),
[SGS-SLAM](https://github.com/ShuhongLL/SGS-SLAM).