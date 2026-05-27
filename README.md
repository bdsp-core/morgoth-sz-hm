
# 🫀 Seizure Detection with Hard Mining and Chewing Artifact Rejection

This repository extends the base **Morgoth** EEG model with two improvements specifically targeting seizure detection:

## 1. Hard Mining

Standard training treats all misclassified examples equally. In seizure detection, the model tends to perform poorly on a small subset of difficult cases — ambiguous ictal patterns, short seizures, and low-amplitude events. Hard mining addresses this by identifying the most frequently misclassified windows during training and up-weighting them in subsequent iterations, forcing the model to focus on the hardest examples rather than easy negatives.

See `hard_mining.py` for the implementation and `train_hard_mining.sh` / `train_hard_mining_2.sh` for training scripts.

## 2. Chewing Artifact Rejection (Few-Shot Learning)

Chewing artifacts are a major source of false positives in automated seizure detection. Their high-frequency, rhythmic muscle activity closely mimics ictal EEG patterns, making them difficult to distinguish without additional context. Because labeled chewing artifact data is scarce, this repository uses a **few-shot learning** approach to train a lightweight auxiliary classifier that identifies and rejects chewing artifacts before seizure classification.

The combined pipeline:
```
EEG window → Morgoth event-level features → Chewing artifact classifier → (if artifact: reject) → Seizure classifier → output
```

See `szcore.py` for the full scoring pipeline and `combine_hm_fsl_results.py` for merging hard mining and few-shot learning outputs.

---

## 🚀 Running Inference

All three modes are controlled by a single script: `run_IIIC_detection_continuous_event_level.sh`

### Step 1 — Set your paths and mode

Open the script and edit the top section:

```bash
# Options: hm_only | chewing_only | combined
MODE="combined"

EEG_DIR="/your/eeg/data/dir"    # directory containing input EDF files
RESULT_DIR="/your/results/dir"  # directory where output CSVs will be saved

HM_MODEL="checkpoints/sz_hm13_1.pth"
CHEWING_MODEL="checkpoints/IIIC_chewing3/sz_fs_chewing.pth"
```

### Step 2 — Choose a mode

| Mode | When to use | Checkpoint(s) used |
|------|------------|-------------------|
| `hm_only` | Standard IIIC classification without artifact rejection | `HM_MODEL` only |
| `chewing_only` | Inspect chewing artifact scores in isolation | `CHEWING_MODEL` only |
| `combined` | Clinical use — chewing artifacts suppressed before classification (recommended) | `HM_MODEL` + `CHEWING_MODEL` |

### Step 3 — Run

```bash
bash run_IIIC_detection_continuous_event_level.sh
```

### Output

Results are saved to `RESULT_DIR/<mode>_pred_1sStep/`, one CSV per input EDF file.

Each CSV contains per-second predictions with 6 class probabilities:

| Column | Class |
|--------|-------|
| 0 | Other |
| 1 | Seizure |
| 2 | LPD |
| 3 | GPD |
| 4 | LRDA |
| 5 | GRDA |

---

# 🧠 Morgoth: Toward Automated EEG Interpretation (For Reference Only)

This repository contains code and tools for running EEG analysis using **Morgoth**, including event-level and EEG-level detection with probabilities for:

- Normal / Abnormal (binary)
- Slowing: No Slowing / Focal Slowing / Generalized Slowing (in the same order as each column in the corresponding output file)
- Burst suppression: No / Burst suppression (binary)
- Spike detection: No / Spike (binary)
- Spike localization: No / Focal Spike / Generalized Spike (in the same order as each column in the corresponding output file)
- IIIC classification: Other / Seizure / LPD / GPD / LRDA / GRDA (in the same order as each column in the corresponding output file)
- IIIC classification with chewing artifact correction (binary per class, uses an auxiliary high-frequency artifact model)
- Sleep staging (5 stages): Awake / N1 / N2 / N3 / REM (in the same order as each column in the corresponding output file)
- Sleep staging (3 stages): Awake / N1 / N2 (in the same order as each column in the corresponding output file)
- Sleep staging (6 stages): Awake / N1 / N2 / N3 / REM / Other (in the same order as each column in the corresponding output file)
- Sleep arousal detection: No / Arousal (binary)
- Vertex wave detection: No / Vertex wave (binary)
- BIRD (Brief Ictal Rhythmic Discharge): No / BIRD (binary)
- BIPD (Bilateral Independent Periodic Discharges): No / BIPD (binary)
- Partial PD (focal periodic discharges with lateralization): No / Left / Right (in the same order as each column in the corresponding output file)
## 📁 Directory Structure

```
├── test_data/                  # EEG data (.mat/.pkl/.edf)
├── checkpoints/                # Pretrained models and checkpoints
├── xxx.py                      # Code files
├── xxx.sh/bat                  # Run model
└── README.md                   # Project overview
```
## 📥 Download Model and Test Data

Before running the code, please download the pretrained model (checkpoints) and test dataset (test_data) from AWS S3 and place them in the appropriate folders:

```bash
aws s3 cp s3://bdsp-opendata-credentialed/morgoth2/models/202605/morgoth checkpoints/ --recursive
```

## ⚙️ Setup

### 1. Create environment (conda recommended) in morgoth folder
### python 3.12 + pytorch 2.4 + CUDA 12.4

```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements.txt / conda install --file requirements.txt

# If you have GPU with cuda driver
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```


### 2. Optional: python 3.11 + pytorch 2.1 + CUDA 12.2 / 12.1
```bash
conda create -n morgoth python=3.12 

conda activate morgoth

pip install -r requirements_cuda122.txt

# If you have GPU with cuda driver
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```


## 🚀 Usage

### Before Start

The model supports raw EEG data in both EDF and MAT formats as input.

This means you can use unaltered clinical recordings directly — no manual preprocessing is required. The model automatically performs all necessary preprocessing steps, including: 

Bandpass filtering, Resampling, Montage, Clipping, Normalization, and Epoching.

⚠️ To enable this pipeline, users must either:

- Ensure the raw EEG file contains sampling rate and channel names, or

- Provide this information explicitly via command-line arguments, or 

- Channel order must follow the standard as specified in the corresponding bash scripts. 

Please refer to the continuous_event_level.sh, discrete_event_level.sh, or EEG_level.sh for example configurations and expected input parameters.

This design ensures a streamlined and reproducible workflow, allowing you to run the model on raw EEG files directly, without requiring prior signal processing expertise.

### 1. Run continuous prediction on event level

See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

```bash
bash continuous_event_level.sh
```

### 2. Run discrete prediction
See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

The data in test_data folder should be processed first.

```bash
bash discrete_event_level.sh
```

### 3. Run EEG level prediction
See the comments at the beginning of the Bash file to understand the meaning of each parameter in the command. You can modify parameters inside each script.

The EEG_level inputs are the event-level output results with 1-second slipping step.

```bash
bash eeg_level.sh
```


## 🏋️‍♂️ Train a Model from Scratch

To train a Morgoth model from scratch using your own data:

### 1. Prepare your dataset in `.h5` for pretraining and `.pkl` for fine tuning 

You should modify the data_provider.py script according to your dataset

```bash
echo password | sudo -S ~/miniconda3/envs/torchenv/bin/python data_provider.py
```

### 2. Run the pretraining script:

```bash
bash run_pretrain.sh
```

### 3. Run the fine-tuning script for event-level:

```bash
bash train_classification.sh
```

### 4. Run the fine-tuning script for EEG-level:

```bash
bash train_EEG_level_head.sh
```

You may modify the script or config file to set the number of epochs, learning rate, batch size, model type, etc.

Make sure you have sufficient GPU memory for large models or long EEG recordings.


## 🔧 Errors and Solutions

### NumPy and Pandas Binary Incompatibility

Reinstall NumPy using Conda with the command:

```bash
conda install numpy=1.26.4
```


## 🚨 Update 

### 2025-06-08

Add layer clip for cuda index out of boundary in EEG_level_head.sh, class CNNTransformerClassifier, def forward

```bash
# check the valid of index
if indices.max() >= x.size(1) or indices.min() < 0:
    indices = torch.clamp(indices, 0, x.size(1) - 1)
```

### 2025-06-10

When encountering pandas and numpy compatibility issues, please reinstall NumPy 1.26.4 using Conda instead of pip.

```bash
conda install numpy=1.26.4
```

### 2025-07-24

#### Running Morgoth on Long EEGs by Segmenting

If you cannnot run Morgoth on long EEG files because of GPU memory limitations. To perform continuous event-level detection, run the following command:

```bash

bash continuous_event_level_longeeg.sh

```
This script will automatically segment the EEG files and batch them through Morgoth. The final output will contain predictions for each segment. In the script, you’ll need to specify the following arguments:

```bash

--data_format edf \                               # format of the raw EEG data 
--segment_duration 600 \                          # duration of each segment in seconds (here, 10 minutes) 
--eeg_dir test_data/longEEG/raw \                 # directory containing the original raw EEG files 
--eval_sub_dir test_data/longEEG/segments_raw \   # directory to store the segmented files

```

continuous_event_level_longeeg.sh included examples in the script for IIIC, spike, slowing, and burst suppression detection. You can follow a similar format for other tasks. Just make sure to keep the --eval_sub_dir path consistent across all commands.

If you need to merge the segment-level results into a full-length prediction, note that you can’t just concatenate them — due to the sliding step, you’ll need to align each segment by padding the predictions before merging.


### 2025-08-02

Wrong code in utils.py resize_array_along_axis0()

Change 

```bash
return arr[indices:,]
```

to 

```bash
return arr[indices,: ]
```


### 2025-09-25

If your GPU driver is ≤ 550, please switch to a different CUDA version (e.g., pytorch-cuda=12.1)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

to

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2025-09-26

It shows a torch._C error: this may be due to the Conda version itself. Newer Conda uses different download sources than older ones, so the CUDA installed by Conda conflicted with the Torch installed by pip. To fix this, I removed torch from requirements.txt and modified the Conda installation command to install Torch with Conda instead. In other words, the installation steps for Conda and pip needed to be swapped.

1. Create and activate the environment
   
```bash
conda create -n morgoth python=3.12 
 ```

```bash
conda activate morgoth
```

2. conda torch cuda

```bash
conda install -y -c pytorch -c nvidia  pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4
```

If 404:

```bash
conda install -y -c pytorch -c nvidia/label/cuda-12.4.127 pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4
```

3. pip others

change requirements.txt to:

```bash
apex==0.9.10dev
botocore==1.34.154
einops==0.8.0
h5py==3.12.1
hdf5storage==0.1.19
mat73==0.65
matplotlib==3.9.2
mne==1.8.0
mne_connectivity==0.7.0
mypy==1.11.2
pandas==1.5.3
pyhealth==1.1.6
pytz==2024.1
scikit-learn==1.5.2
scipy==1.14.1
sympy==1.13.2
tensorboardX==1.8
timm==1.0.11
tqdm==4.66.5
pytest==7.4.4
colorcet==3.1.0
protobuf==3.20.3
numpy==1.26.4
```

```bash
pip install -r requirements.txt 
```

4. If pandas or numpy error

```bash
conda install numpy=1.26.4 
```


## 📬 Contact

For questions, please contact:  
Chenxi Sun – csun8@bidmc.harvard.edu 
cxsun@stanford.edu
