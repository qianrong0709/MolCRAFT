# 处理过程

## 1. 先从 crossdocked_v1.1_rmsd1.0 数据集中提取口袋

**<font color="red">NOTE！</font>**    
***口袋的范围改成8埃，应该是比较好的***

### 使用 extract_pocket.py ：    
`python -m core.datasets.extract_pockets --source data/crossdocked_v1.1_rmsd1.0 --dest data/crossdocked_v1.1_rmsd1.0_pocket8`


## 2. 再计算电子密度

~~很坑，服务器上安装的phenix只能用python2.7，导致 `batch_map_generate.py` 和 `map_to_pointcloud.py` 两个文件的代码都重新更改了一下~~

~~`nohup phenix.python batch_map_generate.py > map_gen.log 2>&1 &`~~

### **有关服务器的使用**    
发现一个问题，服务器以后最好还是用`sbatch`命令跑代码，这样 vpn 断了还能继续跑。

用`sallco` 交互式，vpn一断，代码就停止了。所以改成 `sbatch map_gen.sh`

### **<font color="red">电子密度处理有点复杂</font>**











