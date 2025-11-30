scGPT预训练流程：
1. 先将h5ad数据转换成scGPT预训练输入的格式。转换脚本：build_data.py
参数说明：
    --input-dir  存放所有h5ad文件
    --output-dir 输出目录
    --vocab-file 基因词典，json文件 {"gene1": 0, "gene2":1}
2. 执行预训练脚本
- pretrain.py为模型预训练流程的代码
- run_pretrain_single_gpu.sh  单卡训练脚本，可自行修改或增加模型相关的配置
- dsub_pretrain.sh  多卡训练，武超集群投递作业，其他集群可参照pretrain.sh脚本（torchrun）

