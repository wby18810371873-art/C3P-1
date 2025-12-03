# ========================================
# C³P: Cluster-based Conformal Predictor
# 完整实验代码框架
# ========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, Dict
import pickle
import os


# ========================================
# 1. 数据加载和预处理
# ========================================
def get_cifar100_datasets(data_dir='./data'):
    """获取CIFAR-100数据集，划分为训练/验证/校准/测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    # 主训练集（用于训练模型）
    train_set = datasets.CIFAR100(root=data_dir, train=True,
                                  download=True, transform=transform)

    # 测试集（用于最终评估）
    test_set = datasets.CIFAR100(root=data_dir, train=False,
                                 download=True, transform=transform)

    # 从训练集划分出验证集和校准集
    train_indices, val_cal_indices = train_test_split(
        range(len(train_set)), test_size=0.3, random_state=42
    )
    val_indices, cal_indices = train_test_split(
        val_cal_indices, test_size=0.5, random_state=42
    )

    return {
        'train': Subset(train_set, train_indices),
        'val': Subset(train_set, val_indices),
        'cal': Subset(train_set, cal_indices),
        'test': test_set
    }


# ========================================
# 2. 模型定义（使用ResNet-50）
# ========================================
class ConformalModel(nn.Module):
    """带保形分数计算的模型包装器"""

    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)



# ========================================
# 3. C³P核心算法实现
# ========================================
class C3P:
    """Cluster-based Conformal Predictor"""

    def __init__(self, model: ConformalModel, num_clusters: int = 20):
        """
        Args:
            model: 训练好的模型
            num_clusters: 类别聚类数量
        """
        self.model = model.eval()
        self.num_clusters = num_clusters
        self.cluster_assignments = None  # 类别到簇的映射
        self.cluster_quantiles = {}  # 簇级别的分位数
        self.class_quantiles = {}  # 类别级别的分位数（用于计算聚类误差）
        self.support_set_threshold = None

    def compute_class_scores(self, dataloader: DataLoader, device: str) -> Dict[int, np.ndarray]:
        """计算每个类别的保形分数"""
        scores_by_class = {c: [] for c in range(100)}

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                scores = self.model.get_scores(images, labels)

                # 按类别分组
                for idx, (score, label) in enumerate(zip(scores, labels)):
                    scores_by_class[label.item()].append(score.cpu().item())

        return {c: np.array(scores) for c, scores in scores_by_class.items() if len(scores) > 0}

    def cluster_classes(self, val_scores: Dict[int, np.ndarray]):
        """基于类别分数分布进行K-means聚类"""
        # 计算每个类别的平均分数和方差作为特征
        class_features = []
        valid_classes = []

        for c, scores in val_scores.items():
            if len(scores) >= 10:  # 确保有足够样本
                valid_classes.append(c)
                class_features.append([
                    np.mean(scores),
                    np.std(scores),
                    np.median(scores)
                ])

        class_features = np.array(class_features)

        # K-means聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(class_features)

        # 建立类别到簇的映射
        self.cluster_assignments = {
            c: cluster_labels[i] for i, c in enumerate(valid_classes)
        }

        # 为孤儿类分配单独簇
        orphan_classes = set(range(100)) - set(valid_classes)
        for i, c in enumerate(orphan_classes):
            self.cluster_assignments[c] = self.num_clusters + i

    def compute_quantiles(self, cal_scores: Dict[int, np.ndarray], alpha: float):
        """计算类别和簇级别的分位数"""
        # 类别级别分位数
        for c, scores in cal_scores.items():
            self.class_quantiles[c] = np.quantile(scores, 1 - alpha / 2)

        # 簇级别分位数（取簇内所有类别分位数的中位数）
        cluster_scores = {k: [] for k in range(self.num_clusters)}
        for c, cluster_id in self.cluster_assignments.items():
            if c in cal_scores:
                cluster_scores[cluster_id].append(self.class_quantiles[c])

        for cluster_id, scores in cluster_scores.items():
            if len(scores) > 0:
                self.cluster_quantiles[cluster_id] = np.median(scores)
            else:
                self.cluster_quantiles[cluster_id] = 1.0  # 保守估计

    def construct_prediction_set(self, x: torch.Tensor, alpha: float, delta: float = None) -> Tuple[List[int], float]:
        """为单个样本构建预测集"""
        if delta is None:
            delta = alpha / 2  # 默认过滤误差分配

        with torch.no_grad():
            logits = self.model(x.unsqueeze(0).to(next(self.model.parameters()).device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

        # Step 1: 支持集过滤（Top-K + 分布头部）
        sorted_indices = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_indices])
        support_threshold = 1 - delta
        support_set = sorted_indices[cumsum <= support_threshold].tolist()

        # 确保至少包含delta概率的类别
        if len(support_set) == 0:
            support_set = sorted_indices[:max(1, int(delta * 100))].tolist()

        # Step 2: 对支持集中的类别应用簇分位数
        prediction_set = []
        scores = 1 - probs  # 转换成分数

        for c in support_set:
            if c in self.cluster_assignments:
                cluster_id = self.cluster_assignments[c]
                cluster_quantile = self.cluster_quantiles.get(cluster_id, 1.0)
                if scores[c] <= cluster_quantile:
                    prediction_set.append(c)

        return prediction_set, delta


# ========================================
# 4. 训练流程
# ========================================
def train_model(model: ConformalModel, train_loader: DataLoader,
                val_loader: DataLoader, device: str, epochs: int = 50):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0.0

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))
    return model


# ========================================
# 5. 评估指标
# ========================================
def evaluate_coverage(c3p: C3P, test_loader: DataLoader, alpha: float,
                      device: str, num_trials: int = 10) -> Dict[str, float]:
    """评估覆盖率和预测集大小"""
    coverage_scores = []
    set_sizes = []

    for _ in range(num_trials):
        trial_coverage = []
        trial_sizes = []

        for images, labels in test_loader:
            images = images.to(device)

            for i in range(len(labels)):
                pred_set, _ = c3p.construct_prediction_set(
                    images[i], alpha=alpha
                )

                trial_coverage.append(labels[i].item() in pred_set)
                trial_sizes.append(len(pred_set))

        coverage_scores.append(np.mean(trial_coverage))
        set_sizes.append(np.mean(trial_sizes))

    return {
        'coverage_mean': np.mean(coverage_scores),
        'coverage_std': np.std(coverage_scores),
        'set_size_mean': np.mean(set_sizes),
        'set_size_std': np.std(set_sizes)
    }


# ========================================
# 6. 主实验流程
# ========================================
def main():
    # 超参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    ALPHA = 0.1  # 目标错误率
    NUM_CLUSTERS = 20

    # 数据加载
    datasets = get_cifar100_datasets()
    train_loader = DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=BATCH_SIZE)
    cal_loader = DataLoader(datasets['cal'], batch_size=BATCH_SIZE)
    test_loader = DataLoader(datasets['test'], batch_size=BATCH_SIZE)

    # 训练模型（如果已有预训练模型，可跳过）
    print("Training base model...")
    model = ConformalModel(num_classes=100)
    # model = train_model(model, train_loader, val_loader, DEVICE, epochs=50)
    # 实际训练需要GPU资源，这里注释掉，您可以加载预训练权重
    # model.load_state_dict(torch.load("your_pretrained_model.pth"))

    # 使用PyTorch提供的ResNet预训练权重作为示例
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 100)
    model = model.to(DEVICE)

    # C³P算法流程
    print("\nRunning C³P algorithm...")
    c3p = C3P(model, num_clusters=NUM_CLUSTERS)

    # Step 1: 在验证集上计算类别分数并聚类
    print("  Step 1: Clustering classes...")
    val_scores = c3p.compute_class_scores(val_loader, DEVICE)
    c3p.cluster_classes(val_scores)

    # Step 2: 在校准集上计算分位数
    print("  Step 2: Computing quantiles...")
    cal_scores = c3p.compute_class_scores(cal_loader, DEVICE)
    c3p.compute_quantiles(cal_scores, alpha=ALPHA)

    # Step 3: 在测试集上评估
    print("  Step 3: Evaluating on test set...")
    results = evaluate_coverage(c3p, test_loader, ALPHA, DEVICE)

    # 打印结果
    print("\n" + "=" * 50)
    print(f"C³P Results (α = {ALPHA}):")
    print(f"  Coverage: {results['coverage_mean']:.4f} ± {results['coverage_std']:.4f}")
    print(f"  Avg Set Size: {results['set_size_mean']:.2f} ± {results['set_size_std']:.2f}")
    print(f"  Target Coverage: {1 - ALPHA:.4f}")
    print("=" * 50)


if __name__ == "__main__":

    main()
