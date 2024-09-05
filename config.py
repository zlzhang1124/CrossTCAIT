# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/8/24
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : config.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.0
# @License  : None
# @Brief    : 配置文件

import os
import matplotlib
import platform
import pandas as pd
import random
import numpy as np
import torch
from torch.backends import cudnn
import tensorflow as tf


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 设置tensorflow输出控制台信息：1等级，屏蔽INFO，只显示WARNING + ERROR + FATAL
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用第4个GPU,-1不使用GPU，使用CPU
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 200)
np.set_printoptions(threshold=np.inf)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'  # 保存矢量图中的文本在AI中可编辑


if platform.system() == 'Windows':
    DATA_PATH = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\安中医神经病学研究所合作\data\preprocessed_data"
    DATA_PATH_EXT1 = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\言语特征可重复性\data\preprocessed_data"
    DATA_PATH_EXT2 = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\言语与认知老化实验\言语与认知老化\data\raw_data\audio"
    DATA_PATH_TORGO = None
    DATA_PATH_PREP = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\安中医神经病学研究所合作\analysis\SpontaneousSpeech_embedding\ver2_multiTask\data\audio"
    DATA_PATH_PREP_TORGO = r"F:\Graduate\NeurocognitiveAssessment\认知与声音\安中医神经病学研究所合作\analysis\SpontaneousSpeech_embedding\ver3_CrossTVAIT\data\audio\TORGO"
    font_family = 'Arial'
else:
    DATA_PATH = r"/home/zlzhang/data/WD_PD/data/preprocessed_data"
    DATA_PATH_EXT1 = r"/home/zlzhang/data/言语特征可重复性/data/preprocessed_data"
    DATA_PATH_EXT2 = r"/home/medicaldata/ZZLData/Datasets/audio/言语与认知老化实验/言语与认知老化/data/raw_data/audio"
    DATA_PATH_TORGO = r'/home/medicaldata/ZZLData/Datasets/audio/TORGO'
    DATA_PATH_PREP = r"/home/zlzhang/data/WD_PD/analysis/SpontaneousSpeech_embedding/ver2_multiTask/data/audio"
    DATA_PATH_PREP_TORGO = r"/home/zlzhang/data/WD_PD/analysis/SpontaneousSpeech_embedding/ver3_CrossTVAIT/data/audio/TORGO"
    font_family = 'DejaVu Sans'
    ZH_W2V2_BASE_MODEL = "/home/zlzhang/pretrained_models/huggingface_models/chinese-wav2vec2-base"
    ZH_HUBERT_BASE_MODEL = "/home/zlzhang/pretrained_models/huggingface_models/chinese-hubert-base"
    EN_W2V2_BASE_MODEL = "/home/zlzhang/pretrained_models/huggingface_models/wav2vec2-base-960h"
    EN_HUBERT_BASE_MODEL = "/home/zlzhang/pretrained_models/huggingface_models/hubert-base-ls960"
matplotlib.rcParams["font.family"] = font_family


def setup_seed(seed: int):
    """
    全局固定随机种子
    :param seed: 随机种子值
    :return: None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # https://github.com/NVIDIA/framework-reproducibility/blob/master/doc/d9m/tensorflow_status.md#TF_DETERMINISTIC_OPS
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_USE_FRONTEND'] = '1'  # TF 2.5-2.7复现
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False


rs = 323
setup_seed(rs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")
# 保证性别和年龄匹配的被试ID：HC 50人（23M27F）；WD 50人（23M27F）
# WD和HC的性别无显著性差异：卡方检验，χ2 = 0.00, p = 1.00
# 年龄数据正态分布
# WD和HC的年龄无显著性差异（HC：27.74±4.247；WD：29.96±7.404）：t检验，t = -1.839, p = 0.07>0.05
# WD和HC的男性年龄无显著性差异（HC-M：28.22±3.954；WD-M：31.78±8.480）：t检验，t = -1.827, p = 0.077>0.05
# WD和HC的女性年龄无显著性差异（HC-F：27.33±4.515；WD-F：28.41±6.084）：t检验，t = -0.737, p = 0.465>0.05
# 年龄数据非正态分布（该数据年龄非正态）
# WD和HC的年龄无显著性差异：Mann-Whitney U检验，Z = -1.614, p = 0.107>0.05
# WD和HC的男性年龄无显著性差异：Mann-Whitney U检验，Z = -1.695, p = 0.09>0.05
# WD和HC的女性年龄无显著性差异：Mann-Whitney U检验，Z = -0.712, p = 0.477>0.05
id_used = ["20210614001", "20210615002", "20210621007", "20210621008", "20210621010", "20210621011", "20210622013",
           "20210623014", "20210627017", "20210627018", "20210628020", "20210701034", "20210702037", "20210704042",
           "20210704043", "20210705047", "20210713052", "20210714055", "20210721066", "20210721067", "20210725071",
           "20210725072", "20211010108", "20211010110", "20211016117", '20211016118', '20211107128',  # WD-female
           "20210616004", "20210619006", "20210627019", "20210628021", "20210702036", "20210703039", "20210703040",
           "20210705044", "20210706048", "20210711051", "20210714054", "20210714056", "20210718062", "20210719064",
           "20210721065", "20210724068", "20210726074", "20211010105", "20211010107", "20211010109", "20211017120",
           "20211017121", "20211017122",  # WD-male
           "20210614032", "20210619033", "20210821079", "20210822080", "20210912084", "20210912086", "20210912087",
           "20220112021", "20220112030", "20220113031", "20220113032", "20220114024", "20220114033", "20220115025",
           "20220118035", "20220123036", "20220929006", "20221206038", "20221214043", "20230226084", "20230226085",
           "20230211058", "20230213067", "20230223080", "20230223081", "20230224082", "20230227087",  # HC-female
           "20210614029", "20210820078", "20210927093", "20210927095", "20220108008", "20220111018", "20220112020",
           "20220411044", "20220412045", "20220929007", "20221001016", "20221028026", "20221106032", "20221110033",
           "20230108044", "20230108046", "20230208047", "20230208050", "20230214069", "20230221078", "20230222079",
           "20230226086", "20230227088"]  # HC-male

