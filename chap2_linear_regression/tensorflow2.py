import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers, layers, Model

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决负号 '-' 显示为方块的问题

tf.random.set_seed(42)
np.random.seed(42)

def identity_basis(x):
    ret = np.expand_dims(x, axis=1)
    return ret


def multinomial_basis(x, feature_num=10):
    '''多项式基函数 - 带输入归一化'''
    x = np.expand_dims(x, axis=1)  # shape(N, 1)
    
    # 1. 对输入进行归一化到 [0, 1] 范围
    x_min = np.min(x)
    x_max = np.max(x)
    x_normalized = (x - x_min) / (x_max - x_min + 1e-8)  # 避免除零
    
    # 2. 生成多项式特征
    feat = [x_normalized]
    for i in range(2, feature_num+1):
        feat.append(x_normalized ** i)
    ret = np.concatenate(feat, axis=1)
    return ret

def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)
    width = 1.0 * (centers[1] - centers[0])
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x]*feature_num, axis=1)
    
    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret

def gaussian_basis_adaptive(x, feature_num=10):
    '''高斯基函数 - 宽度自适应版本: 基于中心点距离'''
    centers = np.linspace(0, 25, feature_num)
    
    # 计算相邻中心点的平均距离作为宽度
    if feature_num > 1:
        # 方法A: 使用相邻中心点的平均距离
        avg_distance = (centers[1] - centers[0])  # 均匀分布时就是间距
        width = avg_distance * 1.0  # 可以调整这个系数
    else:
        width = 1.0
    
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x]*feature_num, axis=1)
    
    out = (x - centers) / width
    ret = np.exp(-0.5 * out ** 2)
    return ret

def gaussian_basis_knn(x, feature_num=10, k=5):
    '''高斯基函数 - 宽度自适应版本4: 基于KNN思想'''
    centers = np.linspace(0, 25, feature_num)
    
    # 对于每个中心点，找到最近的k个训练点，用它们的距离决定宽度
    widths = []
    for center in centers:
        # 计算所有点到这个中心的距离
        distances = np.abs(x - center)
        # 取第k近的距离作为宽度（排除距离0的自身点）
        kth_distance = np.sort(distances)[min(k, len(x)-1)]
        widths.append(max(kth_distance, 0.1))  # 设置最小宽度避免除零
    
    widths = np.array(widths)
    
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x]*feature_num, axis=1)
    
    out = (x - centers) / widths
    ret = np.exp(-0.5 * out ** 2)
    return ret

def load_data(filename, basis_func=gaussian_basis_adaptive):
    """载入数据"""
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            # 处理可能的数据格式
            parts = line.strip().split()
            xys.append([float(parts[0]), float(parts[1])])
        xs, ys = zip(*xys)
        xs, ys = np.asarray(xs), np.asarray(ys)
        
        # 保存原始数据用于可视化
        o_x, o_y = xs, ys
        
        # 应用基函数变换
        phi0 = np.expand_dims(np.ones_like(xs), axis=1)
        phi1 = basis_func(xs)
        xs = np.concatenate([phi0, phi1], axis=1)
        
        return (np.float32(xs), np.float32(ys)), (o_x, o_y)

# 3. 定义线性回归模型
class LinearModel(Model):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-0.1, maxval=0.1, dtype=tf.float32
            ),
            trainable=True
        )
    
    @tf.function
    def call(self, x):
        y = tf.squeeze(tf.matmul(x, self.w), axis=1)
        return y

# 4. 评估函数
def evaluate(ys, ys_pred):
    """评估模型，计算标准差（RMSE）"""
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std

# 5. 训练函数

def train_one_step(model, optimizer, xs, ys,lambda_reg=0.01):
    """单步训练"""
    with tf.GradientTape() as tape:
        y_preds = model(xs)
        # 使用RMSE作为损失函数
        loss = tf.reduce_mean(tf.sqrt(1e-12 + (ys - y_preds) ** 2))

    # 计算梯度并更新参数
    grads = tape.gradient(loss, model.w)
    optimizer.apply_gradients([(grads, model.w)])
    return loss

@tf.function
def predict(model, xs):
    """预测函数"""
    y_preds = model(xs)
    return y_preds

# 6. 主训练流程
def main():
    # 加载训练数据
    print("=" * 50)
    print("加载训练数据...")
    (xs_train, ys_train), (o_x_train, o_y_train) = load_data('train.txt', basis_func=gaussian_basis_adaptive)
    print(f"训练数据形状: {xs_train.shape}")
    print(f"训练标签形状: {ys_train.shape}")
    
    # 加载测试数据
    print("\n加载测试数据...")
    (xs_test, ys_test), (o_x_test, o_y_test) = load_data('test.txt', basis_func=gaussian_basis_adaptive)
    print(f"测试数据形状: {xs_test.shape}")
    print(f"测试标签形状: {ys_test.shape}")
    
    # 初始化模型
    ndim = xs_train.shape[1]  # 特征维度
    model = LinearModel(ndim=ndim)
    
    # 设置优化器
    optimizer = optimizers.Adam(learning_rate=0.1)
    
    # 训练模型
    print("\n开始训练...")
    print("=" * 50)
    
    epochs = 2000
    loss_history = []
    
    for epoch in range(epochs):
        loss = train_one_step(model, optimizer, xs_train, ys_train)
        loss_history.append(loss.numpy())
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:4d}, Loss: {loss:.4f}')
    
    # 训练集评估
    print("\n" + "=" * 50)
    print("训练集评估:")
    y_train_pred = predict(model, xs_train).numpy()
    train_std = evaluate(ys_train, y_train_pred)
    print(f'训练集预测值与真实值的标准差: {train_std:.4f}')
    
    # 测试集评估
    print("\n测试集评估:")
    y_test_pred = predict(model, xs_test).numpy()
    test_std = evaluate(ys_test, y_test_pred)
    print(f'测试集预测值与真实值的标准差: {test_std:.4f}')
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    
    # 子图1：训练过程损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (RMSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    # 子图2：预测结果可视化
    plt.subplot(1, 2, 2)
    plt.plot(o_x_train, o_y_train, 'ro', markersize=3, label='训练数据')
    plt.plot(o_x_test, o_y_test, 'go', markersize=3, label='测试数据真实值')
    plt.plot(o_x_test, y_test_pred, 'k-', linewidth=2, label='测试集预测值')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Linear Regression (Test RMSE: {test_std:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, train_std, test_std

# 7. 尝试不同的基函数
def compare_basis_functions():
    """比较不同基函数的效果"""
    basis_functions = {
        'Identity': identity_basis,
        'Multinomial': multinomial_basis,
        'Gaussian': gaussian_basis
    }
    
    results = {}
    
    print("\n" + "=" * 60)
    print("比较不同基函数的效果")
    print("=" * 60)
    
    for name, basis_func in basis_functions.items():
        print(f"\n使用 {name} 基函数:")
        print("-" * 40)
        
        # 加载数据
        (xs_train, ys_train), _ = load_data('train.txt', basis_func=basis_func)
        (xs_test, ys_test), (o_x_test, o_y_test) = load_data('test.txt', basis_func=basis_func)
        
        # 初始化模型
        model = LinearModel(ndim=xs_train.shape[1])
        optimizer = optimizers.Adam(learning_rate=0.1)
        
        # 训练
        for epoch in range(500):
            loss = train_one_step(model, optimizer, xs_train, ys_train)
        
        # 评估
        y_test_pred = predict(model, xs_test).numpy()
        test_std = evaluate(ys_test, y_test_pred)
        results[name] = test_std
        
        print(f'测试集标准差: {test_std:.4f}')
        
        # 可视化
        plt.figure(figsize=(8, 5))
        plt.plot(o_x_test, o_y_test, 'go', markersize=3, label='真实值')
        plt.plot(o_x_test, y_test_pred, 'r-', linewidth=2, label='预测值')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{name} Basis Function (Test RMSE: {test_std:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # 打印比较结果
    print("\n" + "=" * 60)
    print("基函数效果比较:")
    for name, rmse in results.items():
        print(f"{name:12s}: RMSE = {rmse:.4f}")
    print("=" * 60)
    
    return results

# 8. 运行主程序
if __name__ == '__main__':
    # 运行主训练流程
    model, train_rmse, test_rmse = main()
    
    # 可选：比较不同基函数的效果
    print("\n是否要比较不同基函数的效果？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        compare_basis_functions()