# -*- coding: utf-8 -*-
# Copyright (c) 2024. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2024/6/3
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : main.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.0: 2024/6/3
#             First version.重新整理
# @License  : None
# @Brief    : 基于的自发言语嵌入特征模型，用于构音障碍分类


if __name__ == "__main__":
    pass
    # Step1: audio_preprocess.py, 音频与处理，获取所需的音频数据
    # Step2: signal_decomposition.py, 语音信号分解，利用CEEMDAN
    # Step3: calcu_features.py, 特征计算，包括DMFCC/Wav2Vec 2.0/HuBERT
    # Step4: models.py, 本土数据集分类
    # Step5: torgo_clf.py, TORGO开源数据集构音障碍分类
