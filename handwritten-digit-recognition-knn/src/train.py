"""训练 KNN 手写数字识别模型，并在测试集上评估。"""
import argparse
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

try:  # 支持从项目根目录运行 python -m src.train
    from .project_utils import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, load_digit_csv
except ImportError:  # 也支持 python src/train.py
    from project_utils import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, load_digit_csv


def main():
    parser = argparse.ArgumentParser(description="训练 KNN 手写数字识别模型")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="训练 CSV 路径，默认 data/手写数字识别.csv")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH,
                        help="模型保存路径，默认 models/knn_digit_model.joblib")
    parser.add_argument("--neighbors", type=int, default=3,
                        help="KNN 的邻居数量 k，默认 3")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="测试集比例，默认 0.2（20%%）")
    args = parser.parse_args()

    if args.neighbors < 1:
        parser.error("--neighbors 必须大于等于 1")
    if not 0 < args.test_size < 1:
        parser.error("--test-size 必须在 0 和 1 之间")

    # 1. 读取数据：X 为 784 个像素特征，y 为数字标签。
    X, y = load_digit_csv(args.data)
    print(f"数据集形状：{X.shape[0]} 行 × {X.shape[1]} 个像素特征")
    print("各数字样本数量：")
    print(y.value_counts().sort_index().to_string())

    # 2. 归一化：训练图片像素从 0～255 缩放至 0～1。
    #    KNN 依赖距离计算，训练与预测必须使用同样的缩放方式。
    X = X.astype("float32") / 255.0

    # 3. 拆分训练集、测试集。
    #    random_state 固定随机结果，便于复现；stratify=y 尽量维持类别比例。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=1,
        stratify=y,
    )
    print(f"训练集：{X_train.shape[0]} 条；测试集：{X_test.shape[0]} 条")

    # 4. 创建并训练 KNN 分类器。
    #    n_neighbors=3 表示查看距离最近的 3 个训练样本，再投票决定类别。
    model = KNeighborsClassifier(n_neighbors=args.neighbors)
    model.fit(X_train, y_train)

    # 5. 用测试集评估。测试集在训练时没有参与模型拟合。
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nKNN (k={args.neighbors}) 测试集准确率：{accuracy:.4%}")
    print("\n分类报告：")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("混淆矩阵（行=真实类别，列=预测类别；类别顺序 0～9）：")
    print(confusion_matrix(y_test, y_pred, labels=list(range(10))))

    # 6. 保存模型。predict.py 会加载这个文件进行单张图片识别。
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"\n模型已保存：{args.model_out.resolve()}")
    print("提示：模型文件由本机训练生成，默认不会提交到 GitHub。")


if __name__ == "__main__":
    main()
