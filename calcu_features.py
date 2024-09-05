# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/8/24 09:56
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : calcu_features.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V1.3: 2024/8/23
#             增加对音频数据的增强
#             V1.2: 2023/4/9
#             修改嵌入特征获取函数的参数，适配直接音频数据输入
#             V1.1: 2023/2/1
#             1. 添加返回类型;
#             2. 重新匹配数据，保证年龄T检验不显著
#             V1.0 - ZL.Z：2022/8/24 - 2022/8/26
# 		      First version.
# @License  : None
# @Brief    : 特征计算

from config import *
import datetime
import glob
import csv
from pathos.pools import ProcessPool as Pool
import warnings
import librosa
from librosa.core import spectrum
from scipy.stats import skew, kurtosis
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, HubertModel
from transformers import logging
from typing import Optional, Union, List, Tuple
import pathlib
from audiomentations import AddGaussianSNR, AddBackgroundNoise, PolarityInversion
import parselmouth

logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="The loudest and softest part in your sound differ by only.*")


def read_csv(filename):
    """读取csv文件"通过csv.reader()来打开csv文件，返回的是一个列表格式的迭代器，
    可以通过next()方法获取其中的元素，也可以使用for循环依次取出所有元素。"""
    if not filename.endswith(".csv"):  # 补上后缀名
        filename += ".csv"
    data = []  # 所读到的文件数据
    with open(filename, "r", encoding="utf-8-sig") as f:  # 以读模式、utf-8编码打开文件
        f_csv = csv.reader(f)  # f_csv对象，是一个列表的格式
        for row in f_csv:
            data.append(row)
    return data


def audio_augmentation(audio_f: str, bgn_dir: str) -> List[Tuple[str, np.float32]]:
    """
    音频数据增强：每10s分割、随机加高斯噪音（5-40dB的SNR）、
    随机加背景噪音（相对原始音频3-30dB的SNR的家庭环境噪音：ESC-50 dataset）
    :param audio_f: 输入.wav音频文件
    :param bgn_dir: 包含背景噪音wav文件的路径
    :return 音频增强后的语音数据列表
    """
    aug_res = []
    audio_name = os.path.basename(audio_f)
    for i_int in np.arange(0., 60., 10.):  # 每10s分割
        index = int(i_int / 10) + 1
        sound = parselmouth.Sound(audio_f).extract_part(i_int, i_int + 10., parselmouth.WindowShape.RECTANGULAR,
                                                        1.0, False)  # type: parselmouth.Sound
        wav, sr = sound.as_array()[0].astype(np.float32), int(sound.sampling_frequency)
        trans_aug_gauss = AddGaussianSNR(min_snr_db=5., max_snr_db=40., p=1.)
        aug_wav_gauss = trans_aug_gauss(samples=wav, sample_rate=sr)
        trans_aug_bgn = AddBackgroundNoise(sounds_path=bgn_dir, min_snr_in_db=3., max_snr_in_db=30.,
                                           noise_transform=PolarityInversion(), p=1.)
        aug_wav_bgn = trans_aug_bgn(samples=wav, sample_rate=sr)
        aug_res += [(audio_name.replace('.wav', f'-{index}'), wav),
                    (audio_name.replace('.wav', f'-{index}_gauss'), aug_wav_gauss),
                    (audio_name.replace('.wav', f'-{index}_bgn'), aug_wav_bgn)]
    return aug_res


def feat_embedding(audio_file_data: Union[str, os.PathLike, pathlib.PurePath, np.ndarray],
                   pretrained_model_name_or_path: Union[str, os.PathLike, pathlib.PurePath],
                   sr: Optional[int] = 16000) -> np.ndarray:
    """
    基于预训练模型Wav2Vec 2.0/HuBERT获取音频嵌入
    :param audio_file_data: 输入音频文件路径或音频数据
    :param pretrained_model_name_or_path: 预训练模型名称或路径
    :param sr: 音频采样率
    :return: 音频嵌入，2999*768维（60s音频，每秒50*768），np.ndarray[shape=(序列帧长度, 768), dtype=float32]
    原始输出为1*2999*768: torch.FloatTensor[cuda, (batch_size, sequence_length, hidden_size)] torch.float32
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path)
    if "hubert" in pretrained_model_name_or_path:
        model = HubertModel.from_pretrained(pretrained_model_name_or_path)
    else:
        model = Wav2Vec2Model.from_pretrained(pretrained_model_name_or_path)
    model = model.to(device)
    model.eval()  # 仅作test模式
    if isinstance(audio_file_data, (str, os.PathLike, pathlib.PurePath)):
        wav, sr = librosa.load(audio_file_data, sr=None)
    else:
        wav = audio_file_data
    input_values = feature_extractor(wav, sampling_rate=sr, return_tensors="pt").input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        outputs = model(input_values)
        feat = outputs.last_hidden_state  # 模型最后一层输出的隐藏状态序列作为最终的嵌入
    return feat.cpu().numpy()[0]


def feat_dmfcc(imfs: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    计算基于信号分解的MFCC特征:DMFCC
    :param imfs: 分解后的信号分量：对于EMD/CEEMDAN/VMD，为本征模态分量IMFs
    :param sr: 音频采样率
    :return: 39维decompose-MFCC特征，每一列为一个特征向量 np.ndarray[shape=(39, n_frames), dtype=float64]
             将上述13*3维MFCC特征计算为统计特征（均值/标准差/偏度/峰度）np.ndarray[shape=(39*4=156,), dtype=float64]
    """
    stft_imfs = []
    for i_imf in imfs:  # 计算每个IMF的短时傅里叶变换
        i_imf_preem = librosa.effects.preemphasis(i_imf, coef=0.97)  # 预加重，系数0.97
        # NFFT=帧长=窗长=400个采样点(25ms,16kHz),窗移=帧移=2/5窗长=2/5*400=160个采样点(10ms,16kHz),汉明窗
        stft_imfs.append(spectrum.stft(i_imf_preem, n_fft=int(0.025*sr), hop_length=int(0.01*sr), window=np.hamming))
    stft_imfs.reverse()  # IMF1-N频率依次降低，因此需要反转
    stft_imfs_comp = np.array(stft_imfs).reshape(-1, np.array(stft_imfs).shape[-1])  # 频率从低到高合成频谱
    n_fft = 2 * (stft_imfs_comp.shape[0] - 1)  # 频谱合成后NFFT变化
    spectrogram_mag = np.abs(stft_imfs_comp)  # 幅值谱/振幅谱	print(spectrogram_mag.shape)
    spectrogram_pow = 1.0/n_fft * np.square(spectrogram_mag)  # 功率谱/功率谱密度PSD
    energy = np.sum(spectrogram_pow, 0)  # 储存每一帧的总能量
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    fb_mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=26)  # Mel滤波器组的滤波器数量 = 26
    spectrogram_mel = np.dot(fb_mel, spectrogram_pow)  # 计算Mel谱
    spectrogram_mel = np.where(spectrogram_mel == 0, np.finfo(float).eps, spectrogram_mel)
    spec_mel_log = librosa.power_to_db(spectrogram_mel)  # 转换为log尺度
    # 前13个MFCC系数，升倒谱系数22, shape=(n_mfcc, t)=(13, 帧数n_frames), 每一列为一个特征向量
    mfcc_f = librosa.feature.mfcc(S=spec_mel_log, n_mfcc=13, lifter=22)  # log-mel谱DCT之后得到decompose-MFCC
    mfcc_f[0, :] = np.log10(energy)  # 将第0个系数替换成对数能量值
    # print(mfcc_f, mfcc_f.shape)
    mfcc_delta1 = librosa.feature.delta(mfcc_f)  # 一阶差分
    mfcc_delta2 = librosa.feature.delta(mfcc_f, order=2)  # 二阶差分
    decompose_mfcc = np.vstack((mfcc_f, mfcc_delta1, mfcc_delta2))  # 整合成39维特征
    dmfcc_mean = np.mean(decompose_mfcc, axis=1)
    dmfcc_std = np.std(decompose_mfcc, axis=1)
    dmfcc_skew = skew(decompose_mfcc, axis=1)
    dmfcc_kurt = kurtosis(decompose_mfcc, axis=1)
    dmfcc = np.hstack((dmfcc_mean, dmfcc_std, dmfcc_skew, dmfcc_kurt))
    return dmfcc


class GetFeatures:
    """计算基于自发言语任务的各类特征"""
    def __init__(self, datasets_dir: str, save_file: str, feat_name: list = None, decomp_dir=None,
                 emd_md: str = 'CEEMDAN', audio_aug: bool = True, bgn_dir: str = ''):
        """
        初始化
        :param datasets_dir: 输入包含的数据集文件的路径
        :param save_file: 数据特征集及标签的保存文件
        :param feat_name: 待提取的特征名，list类型，仅提取该列表中的特征，默认为None，即提取全部的特征
        :param decomp_dir: 信号分解的数据主路径
        :param emd_md: 分解方法'EMD'/'CEEMDAN'
        :param audio_aug: 是否进行音频增强
        :param bgn_dir: 包含背景噪音wav文件的路径
        """
        if feat_name is None or feat_name == []:
            self.fts = ['dmfcc', 'w2v2', 'hubert']
        else:
            self.fts = feat_name
        assert set(self.fts).issubset({'dmfcc', 'w2v2', 'hubert'}), \
            f'仅接受以下特证名输入：dmfcc/w2v2/hubert'
        audio_f_list = glob.glob(os.path.join(datasets_dir, r'**/*.wav'), recursive=True)
        self.audio_f_list = [i for i in audio_f_list if os.path.basename(i).split('_')[0] in id_used]
        self.save_file = save_file
        self.decomp_dir = decomp_dir
        self.decomposition_num = {'EMD': 6, 'CEEMDAN': 8}
        self.emd_md = emd_md
        self.audio_aug = audio_aug
        self.bgn_dir = bgn_dir

    def get_features(self, audio_file: str = '') -> pd.DataFrame:
        """
        获取对应音频的全部特征
        :param audio_file: 音频文件
        :return: pd.DataFrame，音频的全部特征及其对应标签
        """
        print("---------- Processing %d / %d: %s ----------" %
              (self.audio_f_list.index(audio_file) + 1, len(self.audio_f_list), audio_file))
        label = {'HC': 0, 'WD': 1}[audio_file.split(os.path.sep)[-4]]
        subject_id = os.path.basename(audio_file).split('_')[0]
        task = {"_CookieTheft.wav": "PD", "_si.wav": "SI"}[os.path.basename(audio_file)[11:]]
        if int(subject_id[:6]) >= 202209:
            csv_data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(audio_file)), f'{subject_id}.csv'))
            name = csv_data['name'][0]
            sex = {"男": 1, "女": 0}[csv_data['sex'][0]]
            age = int(csv_data['age'][0])
        else:
            csv_data = read_csv(os.path.join(os.path.dirname(os.path.dirname(audio_file)), f'{subject_id}.csv'))
            name = csv_data[0][0].split("：")[1]
            age = int(csv_data[0][1].split("：")[1])
            if int(subject_id[:6]) <= 202112:
                sex = {"男": 1, "女": 0}[csv_data[0][3].split("：")[1]]
            else:
                sex = {"男": 1, "女": 0}[csv_data[0][2].split("：")[1]]
        if self.audio_aug:
            aud_f_data = audio_augmentation(audio_file, self.bgn_dir)
            feat_i = pd.DataFrame()
            for aud_f_d in aud_f_data:
                imfs_f = os.path.join(self.decomp_dir, {0: 'HC', 1: 'WD'}[label], subject_id, self.emd_md,
                                      f'{aud_f_d[0]}.npy')
                _feat_i = pd.DataFrame({'id': [subject_id], 'name': [name], 'aug': [aud_f_d[0].split('-')[-1]],
                                        'task': [task], 'age': [age], 'sex': [sex], 'label': [label]})
                for i_ft in self.fts:
                    if i_ft == "dmfcc":
                        if not os.path.exists(imfs_f) or np.isnan(np.load(imfs_f)).all():
                            continue
                        ft = pd.DataFrame(
                            [[feat_dmfcc(np.load(imfs_f)[:self.decomposition_num[self.emd_md], :])]],
                            columns=[i_ft])
                    elif i_ft == "w2v2":
                        ft = pd.DataFrame([[feat_embedding(aud_f_d[-1], ZH_W2V2_BASE_MODEL)]], columns=[i_ft])
                    else:
                        ft = pd.DataFrame([[feat_embedding(aud_f_d[-1], ZH_HUBERT_BASE_MODEL)]], columns=[i_ft])
                    _feat_i = pd.concat([_feat_i, ft], axis=1)
                feat_i = pd.concat([feat_i, _feat_i], ignore_index=True)
        else:
            imfs_f = os.path.join(self.decomp_dir, {0: 'HC', 1: 'WD'}[label], subject_id, self.emd_md,
                                  os.path.basename(audio_file).replace(".wav", ".npy"))
            feat_i = pd.DataFrame({'id': [subject_id], 'name': [name], 'task': [task], 'age': [age],
                                   'sex': [sex], 'label': [label]})
            for i_ft in self.fts:
                if i_ft == "dmfcc":
                    if not os.path.exists(imfs_f) or np.isnan(np.load(imfs_f)).all():
                        continue
                    ft = pd.DataFrame([[feat_dmfcc(np.load(imfs_f)[:self.decomposition_num[self.emd_md], :])]],
                                      columns=[i_ft])
                elif i_ft == "w2v2":
                    ft = pd.DataFrame([[feat_embedding(audio_file, ZH_W2V2_BASE_MODEL)]], columns=[i_ft])
                else:
                    ft = pd.DataFrame([[feat_embedding(audio_file, ZH_HUBERT_BASE_MODEL)]], columns=[i_ft])
                feat_i = pd.concat([feat_i, ft], axis=1)
        return feat_i

    def run_parallel(self, n_jobs=None) -> pd.DataFrame:
        """
        并行处理，保存所有特征至本地文件
        :param n_jobs: 并行运行CPU核数，默认为None;若为1非并行，若为-1或None,取os.cpu_count()全部核数,-1/正整数/None类型
        :return: pd.DataFrame，数据的全部特征及其对应标签
        """
        if n_jobs == -1:
            n_jobs = None
        if n_jobs == 1:
            res = []
            for i_subj in self.audio_f_list:
                res.append(self.get_features(i_subj))
        else:
            with Pool(n_jobs) as pool:
                res = pool.map(self.get_features, self.audio_f_list)
        feats_all = pd.DataFrame()
        for _res in res:
            feats_all = pd.concat([feats_all, _res], ignore_index=True)
        feats_all.sort_values(by=['label', 'id'], inplace=True, ignore_index=True)
        feats_all.dropna(how='all', inplace=True)
        # feats_all.drop_duplicates(['name', 'task'], keep='first', inplace=True, ignore_index=True)  # 去掉重复被试数据，仅保留最早测试
        feats_all.drop(columns='name', inplace=True)  # 删除姓名列
        feats_all.to_pickle(self.save_file)
        print(feats_all)
        return feats_all


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(
        f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    # audio_data_prep = os.path.join(current_path, 'data/audio/WDHC/preprocess')
    audio_data_prep = DATA_PATH_PREP
    audio_data_bgn = os.path.join(current_path, 'data/audio/DomesticNoise')
    output_f = os.path.join(current_path, r'data/features/features_WDHC.pkl')
    # decomp_path = os.path.join(os.path.dirname(DATA_PATH_PREP), r'decomposition')
    decomp_path = os.path.join(current_path, r'data/decomposition/WDHC/augmentation')
    res_path = os.path.join(current_path, r"results")

    run_feat_flag = True
    if run_feat_flag:  # 特征提取
        get_f = GetFeatures(audio_data_prep, output_f, ['dmfcc', 'w2v2', 'hubert'],
                            decomp_dir=decomp_path, emd_md='CEEMDAN', audio_aug=True, bgn_dir=audio_data_bgn)
        get_f.run_parallel(1)

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(os.path.join(current_path, r"results/finished.txt"), "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
