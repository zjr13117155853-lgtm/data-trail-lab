"""加载已训练的 KNN 模型，对一张 28×28 图片进行预测。"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

try:
    from .project_utils import DEFAULT_DEMO_PATH, DEFAULT_MODEL_PATH, load_image_as_features
except ImportError:
    from project_utils import DEFAULT_DEMO_PATH, DEFAULT_MODEL_PATH, load_image_as_features


def main():
    parser = argparse.ArgumentParser(description="预测一张 28×28 手写数字图片")
    parser.add_argument("--image", type=Path, default=DEFAULT_DEMO_PATH,
                        help="图片路径，默认 data/demo.png")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="训练好的模型路径，默认 models/knn_digit_model.joblib")
    parser.add_argument("--invert", action="store_true",
                        help="反转黑白像素，适用于白底黑字的图片")
    parser.add_argument("--show", action="store_true",
                        help="使用 Matplotlib 展示输入图片")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{args.model}\n"
            "请先运行：python -m src.train"
        )

    # 1. 图片预处理：灰度化、尺寸检查、可选反色、/255、展开为 784 个特征。
    image, features = load_image_as_features(args.image, invert=args.invert)

    # 2. 读取训练好的模型。
    model = joblib.load(args.model)

    # 3. 对齐训练时的特征列名称，避免 sklearn 的 feature names 警告。
    if hasattr(model, "feature_names_in_"):
        columns = list(model.feature_names_in_)
        if len(columns) != features.shape[1]:
            raise ValueError("模型记录的特征数不是 784，请确认模型与当前项目数据格式一致。")
        features_df = pd.DataFrame(features, columns=columns)
        prediction = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[0]
    else:
        # 兼容没有记录列名的旧模型。
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]

    digit = int(prediction[0])
    best_vote = float(np.max(probabilities))
    print(f"图片：{args.image}")
    print(f"预测数字：{digit}")
    print(f"最近邻投票比例（非校准置信度）：{best_vote:.2%}")
    print("各类别投票比例：")
    for label, prob in zip(model.classes_, probabilities):
        print(f"  {int(label)}：{prob:.2%}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.imshow(image, cmap="gray")
        plt.title(f"Prediction: {digit}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
