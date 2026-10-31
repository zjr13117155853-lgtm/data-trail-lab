"""项目共用的数据路径、数据检查和图片预处理函数。"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# 本文件位于 project_root/src/project_utils.py，因此 parents[1] 就是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "手写数字识别.csv"
DEFAULT_DEMO_PATH = PROJECT_ROOT / "data" / "demo.png"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "knn_digit_model.joblib"
IMAGE_SIZE = (28, 28)
N_FEATURES = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 28 * 28 = 784


def load_digit_csv(csv_path: Path):
    """读取 CSV，并检查标签列与 784 个像素特征列。"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到训练 CSV：{csv_path}\n"
            "请将手写数字识别.csv 放入项目 data/ 目录，或通过 --data 指定路径。"
        )

    df = pd.read_csv(csv_path)
    if df.shape[1] != N_FEATURES + 1:
        raise ValueError(
            f"CSV 应有 {N_FEATURES + 1} 列（1 列标签 + {N_FEATURES} 列像素），"
            f"实际检测到 {df.shape[1]} 列。请检查是否多保存了一列索引，或数据格式是否正确。"
        )

    # 第一列是标签，剩余列是图像像素。
    y = pd.to_numeric(df.iloc[:, 0], errors="raise").astype(int)
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors="raise")

    if X.isna().any().any() or y.isna().any():
        raise ValueError("CSV 中存在缺失值，请先清理数据。")
    if not set(y.unique()).issubset(set(range(10))):
        raise ValueError("标签列应只包含 0～9 的整数。")

    pixel_min = float(X.to_numpy().min())
    pixel_max = float(X.to_numpy().max())
    if pixel_min < 0 or pixel_max > 255:
        raise ValueError(
            f"检测到像素范围为 {pixel_min:g}～{pixel_max:g}，"
            "当前教程按 0～255 的原始像素值进行 /255 归一化。"
        )

    # 确保特征列名是字符串。这样 sklearn 会记录 feature_names_in_，
    # 预测时就可以按相同列名、相同顺序传入，避免特征名称警告。
    X.columns = X.columns.astype(str)
    return X, y


def load_image_as_features(image_path: Path, invert: bool = False):
    """将单张 28×28 图片处理成模型使用的 1×784 特征 DataFrame。"""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到待识别图片：{image_path}")

    # convert('L') 将图片转换为单通道灰度图。
    image = Image.open(image_path).convert("L")
    if image.size != IMAGE_SIZE:
        raise ValueError(
            f"图片必须是 28×28，当前图片尺寸为 {image.width}×{image.height}。"
            "请先将图片裁剪/缩放为 28×28。"
        )

    pixels = np.asarray(image, dtype=np.float32)
    # 有些图片是白底黑字，与 MNIST 常见的黑底白字方向相反，
    # 用户可通过 --invert 显式反转，不在程序中自动猜测。
    if invert:
        pixels = 255.0 - pixels

    # 与训练阶段保持一致：原始像素 0～255 -> 浮点数 0～1。
    pixels = pixels / 255.0
    features = pixels.reshape(1, N_FEATURES)
    return image, features
