import configargparse
from utils import str2bool
from datetime import datetime

def get_parser():
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", is_config_file=True, help="config文件路径")
    parser.add_argument("--seed", type=int, default=20, help="随机种子")

    parser.add_argument('--num_workers', type=int, default=0)

    # 定义训练相应的参数
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument("--log_interval", type=int, default=1)

    # 定义优化器的参数
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 定义学习率变化的参数
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=False)

    # 定义迁移学习相关
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss_type', type=str, default='dann')

    # 定义数据相关
    parser.add_argument('--dataset_name', type=str, default="seed3", help="数据集")
    parser.add_argument("--num_of_class", type=int, default=3, help="定义类别数量, 对于不同数据集类别数不同")
    parser.add_argument('--num_of_subjects', type=int, default=15, help="受试者数量")
    parser.add_argument('--feature_dim', type=int, default=310)

    # TODO SEED和SEED-IV Feature数据集
    parser.add_argument('--session', type=int, default=1, help="定义此次训练的session")
    parser.add_argument("--seed3_path", type=str, default = "D:\\EEG\\SEED\\ExtractedFeatures\\")
    parser.add_argument("--seed4_path", type=str, default = "D:\\EEG\\SEED_IV\\eeg_feature_smooth\\")
    
    # # TODO DEAP和DREAMER Signal数据集
    # parser.add_argument("--deap_path", type=str, default = "E:\\EEG_DataSets\\DEAP")
    # parser.add_argument("--dreamer_path", type=str, default = "E:\\EEG_DataSets\\DREAMER")
    # parser.add_argument("--emotion", type=str, default="valence")

    parser.add_argument("--feature_name", type=str, default="de")
    parser.add_argument("--window_sec", type=int, default=1)
    parser.add_argument("--step_sec", type=int, default=None)

    # 定义是否存储模型
    parser.add_argument('--saved_model', type=str2bool, default=False, help="当前训练过程是否存储模型")
    current_date = datetime.now().strftime("%m%d")
    parser.add_argument("--tmp_saved_path", type=str, default=f"D:\\EEG\\logs\\default\\{current_date}\\")
# <<<<<<<< HEAD:2025/06-26-SDHEDN_ablation/config.py"" """ """
# ========

# >>>>>>>> maxupdbpm:2025/05-28-MetaLearningDG/config.py

    # For PMEEG DBSCAN
    parser.add_argument("--eps", type=float, default=1)
    parser.add_argument("--min_samples", type=int, default=5)
# <<<<<<<< HEAD:2025/06-26-SDHEDN_ablation/config.py

    # HEDN相关
    parser.add_argument("--constraint_loss_weight", type=float, default=0.01, help="约束损失权重")
    parser.add_argument("--num_of_sources", type=int, default=14, help="源域簇的数量")
    parser.add_argument("--momentum", type=float, default=0.9, help="动量参数")

    # Ablation study
    parser.add_argument("--ablation", type=str, default="main", help="Ablation study type")

    parser.add_argument("--num_of_val", type=int, default=3)


    return parser   
# ========
    
    # Meta-learning相关参数
    parser.add_argument('--inner_lr', type=float, default=1e-3, help="内循环学习率")
    parser.add_argument('--n_inner_steps', type=int, default=1, help="内循环迭代次数")
    return parser
# >>>>>>>> maxupdbpm:2025/05-28-MetaLearningDG/config.py
