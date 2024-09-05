# -*- coding: utf-8 -*-
# Copyright (c) 2023. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2023/4/3 22:02 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : torgo_clf.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V1.1: 2024/8/21 - 2024/8/23
#             1. 增加额外实验，与本土数据集的消融与对比实验一致
#             V1.0 - ZL.Z：2023/4/3 - 2023/4/4
# 		      First version.
# @License  : None
# @Brief    : TORGO开源数据集构音障碍分类

from config import *
import signal_decomposition
import calcu_features
import models
import re
import datetime
from pathos.pools import ProcessPool as Pool
import glob
import librosa
import soundfile as sf
import noisereduce as nr
import subprocess
from itertools import chain
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from attention import Attention
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from keras_bert import AdamWarmup
import gc
import time


def database_select(dur_min_thr: float = -np.inf, fig: bool = False):
    """
    选择TORGO数据集中的Restricted和Unrestricted sentences任务数据，并计算各个音频时长的统计量
    :param dur_min_thr: 音频最小阈值，s
    :param fig: 是否绘制数据时长频率直方图
    :return: 已选的数据集描述
    """
    subj_l, sess_l, count_l, dur_sum_l, dur_mean_l, dur_std_l, dur_min_l, dur_max_l, dur_25qua_l, \
    dur_50qua_l, dur_75qua_l, dur_l, wav_num_l = [], [], [], [], [], [], [], [], [], [], [], [], []
    for i_sub in glob.iglob(os.path.join(DATA_PATH_TORGO, r"*/Session*/wav_headMic")):
        subj_l.append(i_sub.split(os.sep)[-3])
        sess_l.append(i_sub.split(os.sep)[-2])
        dur_l_i, wav_num_l_i = [], []
        for j_f in os.listdir(i_sub.replace('wav_headMic', 'prompts')):
            with open(os.path.join(i_sub.replace('wav_headMic', 'prompts'), j_f), mode="r") as f:
                cont = re.sub(r"\[.*\]", "", f.readline()).strip()  # 删除文本内容中的注释及首位空格：[*]
                if (len(cont.split(" ")) > 1) or ('.jpg' in cont):  # 仅保留Restricted和Unrestricted sentences任务的音频
                    wav_f = os.path.join(i_sub, j_f.replace('.txt', '.wav'))
                    if os.path.exists(wav_f) and librosa.get_duration(filename=wav_f) > dur_min_thr:
                        wav_num_l_i.append(j_f.replace('.txt', ''))
                        dur_l_i.append(librosa.get_duration(filename=wav_f))
        wav_num_l.append(wav_num_l_i)
        count_l.append(len(wav_num_l_i))
        dur_l.append(dur_l_i)
        dur_sum_l.append(np.sum(dur_l_i))
        dur_mean_l.append(np.mean(dur_l_i))
        dur_std_l.append(np.std(dur_l_i))
        dur_min_l.append(np.min(dur_l_i))
        dur_max_l.append(np.max(dur_l_i))
        dur_25qua_l.append(np.quantile(dur_l_i, 0.25))
        dur_50qua_l.append(np.quantile(dur_l_i, 0.5))
        dur_75qua_l.append(np.quantile(dur_l_i, 0.75))
    data_info = pd.DataFrame({"subj": subj_l, "sess": sess_l, "count": count_l, "dur_sum": dur_sum_l,
                              "dur_mean": dur_mean_l, "dur_std": dur_std_l, "dur_min": dur_min_l,
                              "dur_max": dur_max_l, "dur_25%": dur_25qua_l, "dur_50%": dur_50qua_l,
                              "dur_75%": dur_75qua_l, "dur": dur_l, "wav_num": wav_num_l, })
    dat_info_sel = pd.merge(data_info[['subj', 'count']].groupby('subj', as_index=False).max(),
                            data_info, on=['subj', 'count'], how='left')  # 仅保留数据量最多的session
    # 添加总数的一行
    dat_info_sel.loc[dat_info_sel.index.max()+1] = ['All', dat_info_sel['count'].sum(), np.NAN,
                                                    np.sum(list(chain.from_iterable(dat_info_sel['dur']))),
                                                    np.mean(list(chain.from_iterable(dat_info_sel['dur']))),
                                                    np.std(list(chain.from_iterable(dat_info_sel['dur']))),
                                                    np.min(list(chain.from_iterable(dat_info_sel['dur']))),
                                                    np.max(list(chain.from_iterable(dat_info_sel['dur']))),
                                                    np.quantile(list(chain.from_iterable(dat_info_sel['dur'])), 0.25),
                                                    np.quantile(list(chain.from_iterable(dat_info_sel['dur'])), 0.5),
                                                    np.quantile(list(chain.from_iterable(dat_info_sel['dur'])), 0.75),
                                                    list(chain.from_iterable(dat_info_sel['dur'])),
                                                    dat_info_sel['wav_num'].to_list()]
    # count=1494, dur_sum=7916.285437s, dur_mean=5.298718s, dur_std=3.334618s,
    # dur_min=0.00825, dur_max=43.155062, dur_25%=3.5575, dur_50%=4.285, dur_75%=5.94375
    # print(dat_info_sel)
    if fig:
        plt.figure(figsize=(8, 6), tight_layout=True)
        plt.xlabel('Duration (s)', fontdict={'family': font_family, 'size': 18})
        plt.ylabel('Count', fontdict={'family': font_family, 'size': 18})
        ax = sns.histplot(dat_info_sel[dat_info_sel['subj'] == 'All']['dur'].to_numpy()[0], kde=True, color='steelblue',
                          line_kws={"lw": 3}, binwidth=1, discrete=True)
        ax.lines[0].set_color('red')
        plt.xticks(fontsize=16, fontproperties=font_family)
        plt.yticks(fontsize=16, fontproperties=font_family)
        for sp in plt.gca().spines:
            plt.gca().spines[sp].set_color('k')
            plt.gca().spines[sp].set_linewidth(1)
        plt.gca().tick_params(direction='in', color='k', length=5, width=1)
        plt.grid(False)
        fig_file = f'results/torgoHistplot/TORGO_histplot.png'
        if not os.path.exists(os.path.dirname(fig_file)):
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.svg'), format='svg', bbox_inches='tight', pad_inches=0.02)
        plt.savefig(fig_file.replace('.png', '.pdf'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    transparent=True)
        plt.savefig(fig_file.replace('.png', '.tif'), dpi=600, bbox_inches='tight', pad_inches=0.02,
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.show()
        plt.close('all')
    return dat_info_sel


def audio_preprocess(audio_file: str, output_folder: str, seg_dur: float = 5.0, normalize=True, denoise=False):
    """
    音频预处理：包括格式转换、分割、降噪、归一化
    :param audio_file: 待处理音频文件
    :param output_folder: 输出文件夹
    :param seg_dur: 分割音频段的持续时长
    :param normalize: 是否进行归一化，幅值归一化至[-1,1]，默认是
    :param denoise: 是否进行降噪，由于降噪对于不同音频效果有差异，默认否
    :return: None
    """
    print(audio_file + "音频数据处理中...")
    if not os.path.exists(audio_file):
        raise FileExistsError(audio_file + "输入文件不存在！")
    aud_dur = librosa.get_duration(filename=audio_file)
    if aud_dur < seg_dur:
        print(f"输入的音频时长小于{seg_dur}s，略过")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    audio_name = os.path.basename(audio_file)
    temp_audio = os.path.join(output_folder, "temp_" + audio_name)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    print(f"----------{audio_name} STEP1: 音频格式转换----------")
    # 调用ffmpeg，将任意格式音频文件转换为.wav文件，pcm有符号16bit,1：单通道,16kHz，不显示打印信息
    subprocess.run("ffmpeg -loglevel quiet -y -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s" %
                   (audio_file, temp_audio), shell=True)
    for i_seg in range(int(aud_dur // seg_dur)):
        wav, sr = librosa.load(temp_audio, sr=None, offset=i_seg*seg_dur, duration=seg_dur)
        if denoise:
            print(f"----------{audio_name} STEP2 ({i_seg + 1}/{int(aud_dur // seg_dur)}): 降噪----------")
            reduced_noise = nr.reduce_noise(y=wav, sr=sr, n_fft=512, prop_decrease=0.9)
        else:
            reduced_noise = wav
        if normalize:
            print(f"----------{audio_name} STEP3 ({i_seg + 1}/{int(aud_dur // seg_dur)}): 归一化----------")
            wav_norm = librosa.util.normalize(reduced_noise)
        else:
            wav_norm = reduced_noise
        if int(aud_dur // seg_dur) == 1:
            output_audio = os.path.join(output_folder, audio_name)
        else:
            output_audio = os.path.join(output_folder, audio_name.replace(".wav", f"-{i_seg + 1}.wav"))
        sf.write(output_audio, wav_norm, sr, subtype="PCM_16")
    if os.path.exists(temp_audio):
        os.remove(temp_audio)


def run_audio_preprocess_parallel(preprocessed_path: str, n_jobs=None, **kwargs):
    """
    并行运行音频预处理
    :param preprocessed_path: 预处理保存数据文件路径
    :param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
    :param kwargs: audio_preprocess参数
    :return: None
    """
    assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
    if n_jobs == -1:
        n_jobs = None
    wav_f_l = []
    _data_info_sel = database_select()
    for ind in _data_info_sel.index:
        if _data_info_sel.loc[ind, 'subj'] != 'All':
            for j_wav in _data_info_sel.loc[ind, 'wav_num']:
                i_sub = _data_info_sel.loc[ind, 'subj']
                i_sess = _data_info_sel.loc[ind, 'sess']
                wav_f_l.append(os.path.join(DATA_PATH_TORGO, f"{i_sub}/{i_sess}/wav_headMic/{j_wav}.wav"))

    def audio_prep(audio_f):
        print("---------- Processing %d / %d: %s ----------" %
              (wav_f_l.index(audio_f) + 1, len(wav_f_l), audio_f))
        output_path = os.path.join(preprocessed_path, audio_f.split(os.sep)[-4])
        audio_preprocess(audio_f, output_path, **kwargs)
    if n_jobs == 1:
        for i_wav in wav_f_l:
            audio_prep(i_wav)
    else:
        with Pool(n_jobs) as pool:
            pool.map(audio_prep, wav_f_l)


class SignalDecomposition(signal_decomposition.SignalDecomposition):
    """语音信号分解"""

    def __init__(self, **kwargs):
        """
        初始化
        :return: None
        """
        super().__init__(**kwargs)

    def decomposition_save(self, audio_f: str = '', res_dir: str = '', method: str = 'CEEMDAN'):
        """
        提取全部信号分量，并保存为本地npy文件
        :param audio_f: 被试主文件夹路径
        :param res_dir: 结果保存路径
        :param method: 分解方法'EMD'/'CEEMDAN'/'VMD'
        :return: None
        """
        try:
            print("---------- Processing %d / %d: ./%s ----------" %
                  (self.audio_f_list.index(audio_f) + 1, len(self.audio_f_list),
                   os.path.relpath(audio_f, os.getcwd())))
        except ValueError:
            pass
        subject_id = os.path.normpath(audio_f).split(os.sep)[-2]
        save_path = os.path.join(res_dir, subject_id, method)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        signal, sr = librosa.load(audio_f, sr=None)
        save_name = os.path.basename(audio_f).replace(".wav", "")
        save_f = os.path.join(save_path, f'{subject_id}_{save_name}.npy')
        if not os.path.exists(save_f):
            imfs_residual = self.get_decomposition(signal, method, show=False)
            np.save(save_f, imfs_residual)
            print(f'{subject_id}_{save_name} npy file saved ({method})')
        else:
            print(f'{subject_id}_{save_name} npy file already exists, skipped ({method})')


class GetFeatures(calcu_features.GetFeatures):
    """计算基于自发言语任务的各类特征"""
    def __init__(self, **kwargs):
        """
        初始化
        """
        super().__init__(**kwargs)
        self.audio_f_list = glob.glob(os.path.join(kwargs.get('datasets_dir'), r'**/*.wav'), recursive=True)

    def get_features(self, audio_file: str = '') -> pd.DataFrame:
        """
        获取对应音频的全部特征
        :param audio_file: 音频文件
        :return: pd.DataFrame，音频的全部特征及其对应标签
        """
        print("---------- Processing %d / %d: %s ----------" %
              (self.audio_f_list.index(audio_file) + 1, len(self.audio_f_list), audio_file))
        subject_id = os.path.normpath(audio_file).split(os.sep)[-2]
        wav_num = os.path.basename(audio_file).replace('.csv', '')
        label = 0 if 'C' in subject_id else 1
        feat_i = pd.DataFrame({'id': [subject_id], 'task': ['Sentences'], 'wav_num': [wav_num], 'label': [label]})
        imfs_f = os.path.join(self.decomp_dir, subject_id, self.emd_md,
                              f'{subject_id}_{os.path.basename(audio_file).replace(".wav", ".npy")}')
        for i_ft in self.fts:
            if i_ft == "dmfcc":
                ft = pd.DataFrame([[calcu_features.feat_dmfcc(np.load(imfs_f)[:self.decomposition_num[self.emd_md], :])]],
                                  columns=[i_ft])
            else:
                wav, sr = librosa.load(audio_file, sr=None)
                if i_ft == "w2v2":
                    ft = pd.DataFrame([[calcu_features.feat_embedding(wav, ZH_W2V2_BASE_MODEL, sr=sr)]], columns=[i_ft])
                else:
                    ft = pd.DataFrame([[calcu_features.feat_embedding(wav, ZH_HUBERT_BASE_MODEL, sr=sr)]], columns=[i_ft])
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
        feats_all.sort_values(by=['label', 'id', 'wav_num'], inplace=True, ignore_index=True)
        feats_all.dropna(inplace=True)
        feats_all.to_pickle(self.save_file)
        print(feats_all)
        return feats_all


class BaselineDMFCCModel(models.BaselineDMFCCModel):
    """基线模型：基于EMD的改进MFCC特征（DMFCC）的SVM模型"""

    def __init__(self, task_name="Sentences", **kwargs):
        """
        初始化
        :param task_name: 使用的言语任务：限制性句子阅读和非限制性句子描述（Sentences）
        """
        super().__init__(task_name=task_name, **kwargs)
        feat_data = pd.read_pickle(kwargs.get("data_file"))  # type: pd.DataFrame
        feat_data = feat_data[feat_data["task"] == "Sentences"].reset_index(drop=True)
        self.subj = np.array(feat_data['id'])
        # print(self.train_data_clf.shape, self.train_label.shape)  # shape=[样本数，特征维数]=[656,156]

    def model_train_evaluate(self, fit: bool = True):
        """
        模型训练
        :param fit: 是否进行训练
        :return: 分类模型
        """
        res = {'acc_cv': [], 'f1_cv': [], 'roc_auc_cv': [], 'sen_cv': [], 'spec_cv': [], 'training_time': []}
        true_l, pred_l, pred_prob_l = [], [], []
        skf = StratifiedGroupKFold(n_splits=self.n_folders, shuffle=True, random_state=rs)  # n折交叉验证，分层采样
        fold_index = 0
        for train_index, test_index in skf.split(self.train_data_clf, self.train_label, self.subj):
            fold_index += 1
            print(f'------- FOLD {fold_index} / {skf.get_n_splits(groups=self.subj)} -------')
            save_model = os.path.join(self.model_clf_save_dir, f'model_{self.model_name}_{fold_index}.h5')
            if fit:
                model_svc = SVC(C=10, gamma='auto', probability=True, random_state=rs)
                # 使用CalibratedClassifierCV概率校准估计器SVC，避免后期predict和predict_proba结果不一致
                model_clf = CalibratedClassifierCV(model_svc, n_jobs=1)  # 仅校准最优参数对应的模型
                start_t = time.time()
                model_clf.fit(self.train_data_clf[train_index], self.train_label[train_index])
                res['training_time'].append(time.time() - start_t)
                joblib.dump(model_clf, save_model)  # 保存模型
            else:
                if os.path.exists(save_model):  # 存在已训练模型且设置加载，
                    print("----------加载分类模型：{}----------".format(save_model))
                    model_clf = joblib.load(save_model)  # 加载已训练模型
                    res['training_time'].append(np.NAN)
                else:
                    raise FileNotFoundError("分类模型不存在，无法评估，请先训练")
            y_preds = model_clf.predict(self.train_data_clf[test_index])
            acc = accuracy_score(self.train_label[test_index], y_preds)
            y_pred_proba = model_clf.predict_proba(self.train_data_clf[test_index])
            res['acc_cv'].append(acc * 100)
            true_l += list(self.train_label[test_index])
            pred_l += list(y_preds)
            pred_prob_l += list(y_pred_proba)
        precision, recall, f1_score, support = precision_recall_fscore_support(true_l, pred_l, average='binary')
        sen = recall
        roc_auc = roc_auc_score(true_l, np.array(pred_prob_l)[:, 1])
        spec = models.scorer_sensitivity_specificity(true_l, pred_l)
        res['f1_cv'] = f1_score * 100
        res['sen_cv'] = sen * 100
        res['roc_auc_cv'] = roc_auc * 100
        res['spec_cv'] = spec * 100
        print(f"CV Accuracy: {np.mean(res['acc_cv']):.2f}±{np.std(res['acc_cv']):.2f}")
        print(f"CV F1 score: {res['f1_cv']:.2f}")
        print(f"CV Sensitivity (Recall): {res['sen_cv']:.2f}")
        print(f"CV Specificity: {res['spec_cv']:.2f}")
        print(f"CV ROC-AUC: {res['roc_auc_cv']:.2f}")
        return res


class BaselineEmbeddingModel(models.BaselineEmbeddingModel):
    """基线模型：基于Embedding（w2v2+hubert）的BiLSTM+Attention模型"""

    def __init__(self, task_name="Sentences", **kwargs):
        """
        初始化
        :param task_name: 使用的自发言语任务：限制性句子阅读和非限制性句子描述（Sentences）
        """
        super().__init__(task_name=task_name, **kwargs)
        feat_data = pd.read_pickle(kwargs.get("data_file"))  # type: pd.DataFrame
        feat_data = feat_data[feat_data["task"] == "Sentences"].reset_index(drop=True)
        self.subj = np.array(feat_data['id'])
        # print(self.train_data_clf.shape, self.train_label.shape)  # shape=[样本数，时间序列帧数，特征维数]=[656,249,1536]

    def model_train_evaluate(self, fit: bool = True):
        """
        模型训练
        :param fit: 是否进行训练
        :return: 交叉验证结果
        """
        res = {'acc_cv': [], 'f1_cv': [], 'roc_auc_cv': [], 'sen_cv': [], 'spec_cv': [], 'training_time': []}
        true_l, pred_l, pred_prob_l = [], [], []
        skf = StratifiedGroupKFold(n_splits=self.n_folders)
        es_callback = EarlyStopping(monitor='loss', patience=10)
        fold_index = 0
        for train_index, test_index in skf.split(self.train_data_clf, self.train_label, self.subj):
            fold_index += 1
            print(f'------- FOLD {fold_index} / {skf.get_n_splits(groups=self.subj)} -------')
            save_model = os.path.join(self.model_clf_save_dir, f'model_{self.model_name}_{fold_index}.h5')
            log_dir = os.path.join(self.model_clf_save_dir, f'logs/model_{self.model_name}_{fold_index}')
            if fit:
                model = self.model_create()
                cp_callback = ModelCheckpoint(filepath=save_model, save_best_only=True, monitor="loss")
                callbacks = [es_callback, cp_callback, TensorBoard(log_dir)]
                start_t = time.time()
                model.fit(self.train_data_clf[train_index], self.train_label[train_index],
                          batch_size=self.batch_size, epochs=self.epochs, verbose=0,
                          # validation_data=(self.train_data_clf[test_index], self.train_label[test_index]),
                          shuffle=True, callbacks=callbacks)
                res['training_time'].append(time.time() - start_t)
            else:
                if os.path.exists(save_model):  # 存在已训练模型且设置加载，
                    print("----------加载分类模型：{}----------".format(save_model))
                    res['training_time'].append(np.NAN)
                else:
                    raise FileNotFoundError("分类模型不存在，无法评估，请先训练")
            model_best = load_model(save_model, custom_objects={'Attention': Attention, 'AdamWarmup': AdamWarmup,
                                                                'MixPooling1D': models.MixPooling1D})
            with tf.device("cpu:0"):  # 防止一次性测试的数据过大导致GPU的OOM
                try:  # 标签为1的概率
                    y_pred_proba = model_best(self.train_data_clf[test_index]).numpy()[:, 1]
                except IndexError:
                    y_pred_proba = model_best(
                        self.train_data_clf[test_index]).numpy().flatten()
            y_pred_label = (y_pred_proba > 0.5).astype("int32")
            acc = accuracy_score(self.train_label[test_index], y_pred_label)
            res['acc_cv'].append(acc * 100)
            true_l += list(self.train_label[test_index])
            pred_l += list(y_pred_label)
            pred_prob_l += list(y_pred_proba)
            # print(acc)
            if fit:
                del model
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                setup_seed(rs)
                gc.collect()
        precision, recall, f1_score, support = precision_recall_fscore_support(true_l, pred_l, average='binary')
        sen = recall
        roc_auc = roc_auc_score(true_l, pred_prob_l)
        spec = models.scorer_sensitivity_specificity(true_l, pred_l)
        res['f1_cv'] = f1_score * 100
        res['sen_cv'] = sen * 100
        res['roc_auc_cv'] = roc_auc * 100
        res['spec_cv'] = spec * 100
        print(f"CV Accuracy: {np.mean(res['acc_cv']):.2f}±{np.std(res['acc_cv']):.2f}")
        print(f"CV F1 score: {res['f1_cv']:.2f}")
        print(f"CV Sensitivity (Recall): {res['sen_cv']:.2f}")
        print(f"CV Specificity: {res['spec_cv']:.2f}")
        print(f"CV ROC-AUC: {res['roc_auc_cv']:.2f}")
        print(f"CV Training Time/s: {np.mean(res['training_time']):.2f}±{np.std(res['training_time']):.2f}")
        return res


class CrossTCAInceptionTimeModel(models.CrossTCAInceptionTimeModel, BaselineEmbeddingModel):
    """基于Embedding（w2v2+hubert）的跨时间、跨通道的多头注意力机制InceptionTime模型"""

    def __init__(self, **kwargs):
        """
        初始化
        """
        super().__init__(**kwargs)


def model_compare(data_file: str = "", task_list=None, model_dict: dict = None,
                  perf_comp_f: str = "", mode: str = 'w', fit: bool = True, load_data=True) -> pd.DataFrame:
    """
    模型比较
    :param data_file: 数据集文件
    :param task_list: 使用的言语任务：结构化（PD）和非结构化（SI）
    :param model_dict: 模型名、模型类字典
    :param perf_comp_f: 模型性能比较的csv文件
    :param mode: csv保存文件模式
    :param fit: 是否进行训练
    :param load_data: 是否加载之前模型已评估好的结果直接获取指标数据
    :return: 各模型各配置的交叉验证结果
    """
    if task_list is None:
        task_list = ["Sentences"]
    if load_data:
        df_res = pd.read_csv(perf_comp_f)
    else:
        task_l, conf_l, acc_l, f1_l, sen_l, spe_l, auc_l, trt_l = [], [], [], [], [], [], [], []
        for tn in task_list:
            for ml in model_dict:
                print(f"------- Running {ml} of {tn} task -------\n")
                _model = model_dict[ml](data_file=data_file, model_name=ml, task_name=tn)
                res = _model.model_train_evaluate(fit=fit)
                task_l.append(tn)
                conf_l.append(ml)
                acc_l.append(f"{np.mean(res['acc_cv']):.2f} ({np.std(res['acc_cv']):.2f})")
                f1_l.append(f"{res['f1_cv']:.2f}")
                sen_l.append(f"{res['sen_cv']:.2f}")
                spe_l.append(f"{res['spec_cv']:.2f}")
                auc_l.append(f"{res['roc_auc_cv']:.2f}")
                trt_l.append(f"{np.mean(res['training_time']):.2f} ({np.std(res['training_time']):.2f})")
        df_res = pd.DataFrame({"Task": task_l, "Feature-Model Config": conf_l, "Acc/%": acc_l, "F1/%": f1_l,
                               "Sen/%": sen_l, "Spe/%": spe_l, "AUC/%": auc_l, "Training Time/s": trt_l})
        with open(perf_comp_f, mode=mode) as f:  # 文件不存在时创建文件，文件已经存在时跳过列标题
            df_res.to_csv(f, header=f.tell() == 0, index=False)
    print(df_res)
    return df_res


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    res_path = os.path.join(current_path, r"results")
    preprocessed_data = os.path.join(current_path, "data/audio/TORGO")
    # preprocessed_data = DATA_PATH_PREP_TORGO
    decomp_path = os.path.join(current_path, r'data/decomposition/TORGO')
    feat_f_torgo = os.path.join(current_path, r'data/features/features_TORGO.pkl')
    pcf_torgo_comp = os.path.join(res_path, r"TORGO_comp.csv")

    run_explore_flag = False
    if run_explore_flag:  # 数据分布探索与选择
        # count=1494, dur_sum=7916.285437s, dur_mean=5.298718s, dur_std=3.334618s,
        # dur_min=0.00825, dur_max=43.155062, dur_25%=3.5575, dur_50%=4.285, dur_75%=5.94375
        info_data = database_select(fig=True)  # 数据集总览
    run_prep_flag = False
    if run_prep_flag:  # 获取数据集预处理
        run_audio_preprocess_parallel(preprocessed_data, n_jobs=-1, normalize=True, denoise=True, seg_dur=5.0)
    run_decomp_flag = False
    if run_decomp_flag:  # 信号分解
        SignalDecomposition(audio_aug=False).run_parallel(preprocessed_data, decomp_path, 'CEEMDAN', n_jobs=50)
    run_feat_flag = False
    if run_feat_flag:  # 特征提取
        GetFeatures(datasets_dir=preprocessed_data, save_file=feat_f_torgo, feat_name=['dmfcc', 'w2v2', 'hubert'],
                    decomp_dir=decomp_path, emd_md='CEEMDAN').run_parallel(1)
    run_compare_base_flag = True
    if run_compare_base_flag:  # CTCAIT模型与基线比较
        model_d = {"DMFCC-CEEMDAN": BaselineDMFCCModel, "BiLSTM-Attention": BaselineEmbeddingModel,
                   "CTCAIT": CrossTCAInceptionTimeModel, }
        model_compare(feat_f_torgo, ["Sentences"], model_d, pcf_torgo_comp, mode='w', fit=False, load_data=True)

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

