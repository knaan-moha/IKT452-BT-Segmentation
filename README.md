# IKT452-BT-Segmentation

This project presents a comparative study of **U-Net** and **DeepLabV3** architectures for **brain tumor segmentation** from MRI images, as part of the IKT452-G Computer Vision course at the University of Agder.

Both models use a **ResNet-50 encoder** pretrained on ImageNet, implemented using the `segmentation_models.pytorch` library.

---

## 📁 Project Structure

```
├── Models/           # U-Net and DeepLabV3 implementations
├── figures/          # Plots and segmented output images
├── utils/            # Preprocessing, metrics, and data loading functions
├── requirements.txt  # Python dependencies
├── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/knaan-moha/IKT452-BT-Segmentation.git
cd IKT452-BT-Segmentation
```

### 2. Set Up a Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training and Evaluation

The model training and evaluation are implemented in Jupyter notebooks:

- [`Models/unet-model.ipynb`](Models/unet-model.ipynb) – Train and evaluate U-Net
- [`Models/deeplabv3.ipynb`](Models/deeplabv3.ipynb) – Train and evaluate DeepLabV3

---

## 🧠 Models

- `U-Net`: Excellent for full tumor region coverage.
- `DeepLabV3`: Strong performance with cleaner, precise boundaries.


---
## 🗂️ Dataset

The dataset used in this project [Brain Tumor Segmentation dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation), is available on Kaggle. The dataset contains 3,064 brain MRI images along with their corresponding binary masks for tumor segmentation.

---

## 📊 Metrics Used

- Accuracy (%)
- Precision (%)
- Recall (%)
- Dice Score (%)
- IoU (%)
- Inference Time (s)

---

## 🧪 Tools & Libraries

- Python 3.9
- PyTorch
- segmentation_models.pytorch
- Albumentations
- Matplotlib, NumPy, OpenCV

---

## 👤 Author

**Zekaria Mohamed**  
