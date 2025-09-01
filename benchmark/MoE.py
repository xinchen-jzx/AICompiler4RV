import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F

class MoEWeightAnalyzer:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-v0.1"):
        """
        初始化MoE权重分析器
        
        Args:
            model_name: HuggingFace模型名称
        """
        self.model_name = model_name
        self.model = None
        self.expert_weights = {}
        
    def load_model(self):
        """加载MoE模型"""
        print(f"Loading model: {self.model_name}")
        # 注意：实际使用时可能需要调整设备映射和量化设置
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
    def extract_expert_weights(self, layer_idx=None):
        """
        提取指定层的专家权重
        
        Args:
            layer_idx: 要分析的层索引，None表示分析所有层
        """
        self.expert_weights = {}
        
        for name, param in self.model.named_parameters():
            # 根据不同模型调整专家权重的识别模式
            if "experts" in name and "weight" in name:
                parts = name.split(".")
                
                # 提取层索引和专家索引
                try:
                    if "block" in name or "layer" in name:
                        layer_num = int([p for p in parts if p.isdigit()][0])
                        expert_num = int([p for p in parts if "expert" in parts[parts.index(p)-1]][0])
                        
                        if layer_idx is None or layer_num == layer_idx:
                            key = f"layer_{layer_num}_expert_{expert_num}"
                            self.expert_weights[key] = param.detach().cpu()
                except:
                    continue
                    
        print(f"Extracted {len(self.expert_weights)} expert weight matrices")
        return self.expert_weights
    
    def compute_weight_similarity(self, method="cosine"):
        """
        计算专家权重之间的相似性
        
        Args:
            method: 相似性度量方法 ("cosine", "l2", "correlation")
        """
        if not self.expert_weights:
            raise ValueError("No expert weights extracted. Run extract_expert_weights first.")
        
        # 将权重展平为向量
        weight_vectors = {}
        for name, weight in self.expert_weights.items():
            weight_vectors[name] = weight.flatten().numpy()
        
        names = list(weight_vectors.keys())
        vectors = np.array([weight_vectors[name] for name in names])
        
        # 计算相似性矩阵
        if method == "cosine":
            similarity_matrix = cosine_similarity(vectors)
        elif method == "l2":
            # 转换L2距离为相似性
            distances = squareform(pdist(vectors, metric='euclidean'))
            similarity_matrix = 1 / (1 + distances)
        elif method == "correlation":
            similarity_matrix = np.corrcoef(vectors)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return similarity_matrix, names
    
    def analyze_weight_statistics(self):
        """分析权重的统计特性"""
        stats = {}
        
        for name, weight in self.expert_weights.items():
            weight_np = weight.numpy()
            stats[name] = {
                'mean': np.mean(weight_np),
                'std': np.std(weight_np),
                'norm': np.linalg.norm(weight_np),
                'sparsity': np.sum(np.abs(weight_np) < 1e-6) / weight_np.size,
                'shape': weight.shape
            }
        
        return stats
    
    def compute_activation_patterns(self, inputs, num_samples=100):
        """
        分析不同输入下的专家激活模式
        
        Args:
            inputs: 输入数据
            num_samples: 采样数量
        """
        # 这需要根据具体模型实现
        # 主要是hook中间层的router输出
        pass
    
    def visualize_similarity_matrix(self, similarity_matrix, names, save_path=None):
        """可视化相似性矩阵"""
        plt.figure(figsize=(12, 10))
        
        # 创建热力图
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        sns.heatmap(
            similarity_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0.5,
            square=True,
            xticklabels=names,
            yticklabels=names,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Expert Weight Similarity Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_singular_values(self):
        """分析权重矩阵的奇异值分解"""
        svd_analysis = {}
        
        for name, weight in self.expert_weights.items():
            if len(weight.shape) == 2:  # 只分析2D权重矩阵
                U, S, V = torch.svd(weight)
                svd_analysis[name] = {
                    'singular_values': S.numpy(),
                    'effective_rank': torch.sum(S > 1e-3).item(),
                    'top_10_energy': torch.sum(S[:10]**2) / torch.sum(S**2)
                }
        
        return svd_analysis
    
    def compute_cka_similarity(self, layer_outputs1, layer_outputs2):
        """
        计算CKA (Centered Kernel Alignment) 相似性
        用于比较不同专家的表示空间
        """
        def center_gram_matrix(K):
            n = K.shape[0]
            one_n = np.ones((n, n)) / n
            return K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # 计算Gram矩阵
        K1 = layer_outputs1 @ layer_outputs1.T
        K2 = layer_outputs2 @ layer_outputs2.T
        
        # 中心化
        K1_c = center_gram_matrix(K1)
        K2_c = center_gram_matrix(K2)
        
        # 计算CKA
        cka = np.trace(K1_c @ K2_c) / np.sqrt(np.trace(K1_c @ K1_c) * np.trace(K2_c @ K2_c))
        
        return cka

# 使用示例
def main():
    # 1. 初始化分析器
    analyzer = MoEWeightAnalyzer("mistralai/Mixtral-8x7B-v0.1")
    
    # 2. 加载模型
    analyzer.load_model()
    
    # 3. 提取专家权重（分析第10层）
    analyzer.extract_expert_weights(layer_idx=10)
    
    # 4. 计算相似性
    similarity_matrix, names = analyzer.compute_weight_similarity(method="cosine")
    
    # 5. 可视化
    analyzer.visualize_similarity_matrix(similarity_matrix, names, "expert_similarity.png")
    
    # 6. 统计分析
    stats = analyzer.analyze_weight_statistics()
    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Mean: {stat['mean']:.4f}")
        print(f"  Std: {stat['std']:.4f}")
        print(f"  Norm: {stat['norm']:.4f}")
        print(f"  Sparsity: {stat['sparsity']:.2%}")
    
    # 7. 奇异值分析
    svd_results = analyzer.analyze_singular_values()
    for name, result in svd_results.items():
        print(f"\n{name}:")
        print(f"  Effective rank: {result['effective_rank']}")
        print(f"  Top-10 energy: {result['top_10_energy']:.2%}")

if __name__ == "__main__":
    main()