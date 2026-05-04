# 代码理解

## bfn4sbdd.py 主要函数解释

### interdependency_modeling()

* forward推理核心模块 
基于当前时间步 t 的 belief 表达 θ（θ_h, μ_pos），输出目标分布的预测参数 —— 坐标预测值 μ 和原子类别概率 logits
*
ligand输入特征



### 批次是怎么构建的，为什么要把特征坐标都拼接到一起

## 🔥 uni_transformer.py 神经网络架构

![Screenshot 2025-11-05 at 12.53.45 PM.png](Screenshot%202025-11-05%20at%2012.53.45%20PM.png)


![Screenshot 2025-11-05 at 12.59.02 PM.png](Screenshot%202025-11-05%20at%2012.59.02%20PM.png)

![Screenshot 2025-11-05 at 7.14.58 PM.png](Screenshot%202025-11-05%20at%207.14.58%20PM.png)

### 一些参数不是很理解

>***self.num_r_gaussian = num_r_gaussian*** <br> 用于 将距离特征 r 映射为高维连续特征表示 的维度数量，常用于图神经网络中对连续距离的编码，特别是在分子建模中

![Screenshot 2025-11-06 at 8.58.48 AM.png](Screenshot%202025-11-06%20at%208.58.48%20AM.png)

![Screenshot 2025-11-06 at 9.59.51 AM.png](Screenshot%202025-11-06%20at%209.59.51%20AM.png)






