# **version1. 代码修改**
***v1版本，不加ligand ED的attention***   
***点云不是每一点都进行embedding，是局部点云整体embedding，得到一个向量***   
>v1版本主要是一个快速实现版本，看是否有初步的效果，评价指标需要再想一下
>>v2版本会加入ligand ED的attention    
>>v3版本会尝试对点云的逐点进行embedding，再做attention  
>>v4版本会在v2的基础上决定，如果ligand的attention有效果再做

## uni_transformer.py

#### 增加 `ED2HAttLayer` 层， 用ed做cross-attention, 更新 h 特征

#### 更改 `AttentionLayerO2TwoUpdateNodeGeneral` ，把 ed 层加进去

#### 更新 `UniTransformerO2TwoUpdateGeneral` 主架构，也是加入 ed 层

