# Camouflaged Object Detection in Images & Video using SINet

**EE5340 Video Analytics — Course Project (2026), IIT Guwahati**
Faculty: **Prof. Prithwijit Guha** · Department: SiSAS

| Member | Roll No. |
|---|---|
| Akash Shukla | 264152017 |
| Abdul Quadir | 264152001 |
| Sanjeev Singh | 264152013 |

🌐 **Project report (webpage):** https://akashshukla-github.github.io/SINet-COD-VideoAnalytics/
📊 **Presentation:** [`assets/ppt/SINet_COD_Presentation.pptx`](assets/ppt/SINet_COD_Presentation.pptx)
▶️ **Run in Kaggle:** [SINet_ObjTrack.ipynb](https://www.kaggle.com/code/sanjeevsingh5459/sinet-objtrack)
🎥 **Video demonstration / PPT recording:** linked in Section 9 of the webpage

---

## What this project is

We study **SINet** (Search & Identification Network, Fan et al., CVPR 2020), which segments objects that blend into their background. We then extend it from single images to **video**.

Our contributions on top of the original repository:
1. **Workflow extraction:** the complete SINet data flow, with tensor shapes and equations, reconstructed from the README diagrams and the source code (webpage Section 3).
2. **Code-vs-paper verification:** three differences documented (Gaussian kernel parameters in Search Attention, the unused ReLU in `BasicConv2d`, and the corrected PDC index range).
3. **`code/test_images.py`:** inference that runs on current PyTorch (the original `MyTest.py` relies on the removed `scipy.misc.imsave`), with MAE / IoU / F_β and auto-saved result and failure panels.
4. **`code/video_inference.py`:** frame-wise SINet with temporal EMA smoothing, connected-component mask, bounding box and centroid tracking, producing `overlay.mp4`, `mask.mp4` and `track.csv`.

## Repository structure

```
├── index.html                      # Project report webpage (GitHub Pages)
├── SINet_Colab_Demo.ipynb          # One-click Colab: weights → image results → video results
├── assets/
│   ├── img/                        # Figures (from the SINet repository, credited)
│   ├── js/tex-svg.js               # MathJax (local copy, renders the equations)
│   └── ppt/SINet_COD_Presentation.pptx
├── results/                        # Our outputs (shown automatically on the webpage)
└── code/
    ├── Src/                        # Original SINet model code (Fan et al.)
    ├── EvaluationTool/             # Original MATLAB evaluation toolbox
    ├── MyTest.py, MyTrain.py       # Original scripts
    ├── test_images.py              # OURS — modern inference + metrics
    └── video_inference.py          # OURS — video extension
```

## Setup & Installation Guide

### Option A — Google Colab (recommended, free GPU)
Open the Colab link above, choose **Runtime → Change runtime type → T4 GPU**, then **Run all**. The notebook:
1. clones this repo and downloads the pretrained `SINet_40.pth` and the test set (from the official SINet Google Drive links);
2. runs `test_images.py` on COD10K-test → `results/our_metrics.json`, `results/panel_*.png`, `results/failure_*.png`;
3. runs `video_inference.py` on an uploaded `input.mp4` → `results/video/…` plus a track plot;
4. downloads `results.zip`. Unzip it into this repo's `results/` folder and push; the webpage picks the files up automatically.

### Option B — Local
```bash
git clone https://github.com/akashshukla-github/SINet-COD-VideoAnalytics.git
cd SINet-COD-VideoAnalytics/code
pip install torch torchvision opencv-python numpy scipy
# download SINet_40.pth (pretrained) and the test set from the official links:
#   https://github.com/DengPingFan/SINet#22-usage
python test_images.py --model_path SINet_40.pth --data_root ./Dataset/TestDataset --dataset COD10K --limit 300
python video_inference.py --video input.mp4 --model_path SINet_40.pth --out_dir ../results/video
```
`--cpu` forces CPU (slow, but it works). Video options: `--alpha` (EMA weight, default 0.6), `--tau` (threshold, default 0.5), `--max_frames`.

## Results (from the SINet authors, COD10K-test)

| Model | S_α ↑ | E_φ ↑ | F_β^w ↑ | MAE ↓ |
|---|---|---|---|---|
| CPD (best SOD baseline) | 0.747 | 0.770 | 0.508 | 0.059 |
| EGNet | 0.737 | 0.779 | 0.509 | 0.056 |
| **SINet (paper)** | **0.771** | **0.806** | **0.551** | **0.051** |
| SINet ◇ (released weights) | 0.776 | 0.864 | 0.631 | 0.043 |

Our own run and the video results are on the webpage (Sections 5.3–5.4).

## Credits & License

The SINet model code (`code/Src`, `MyTest.py`, `MyTrain.py`), the evaluation toolbox and the figures in `assets/img` are by **Deng-Ping Fan, Ge-Peng Ji, Guolei Sun, Ming-Ming Cheng, Jianbing Shen and Ling Shao**: https://github.com/DengPingFan/SINet. They are reused here for **academic, non-commercial purposes** under the original license terms, which are reproduced in [`LICENSE_SINet.txt`](LICENSE_SINet.txt). The COD10K dataset is for non-commercial use only.

```bibtex
@inproceedings{fan2020Camouflage,
  title={Camouflaged Object Detection},
  author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
