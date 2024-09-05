# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2021/12/23 15:49 
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : signal_decomposition.py
# @Software : Python3.6; PyCharm; Windows10 / Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M / 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090
# @Version  : V2.1 - ZL.Z：2023/3/2 - 2023/3/3
# 		      修改分解参数
#             V2.0 - ZL.Z：2022/2/4
# 		      删除LMD方法.
#             V1.0 - ZL.Z：2021/12/23 - 2021/12/27
# 		      First version.
# @License  : None
# @Brief    : 语音信号分解：分别利用EMD/CEEMDAN/VMD

from config import *
import datetime
import psutil
import glob
import warnings
import matplotlib.pyplot as plt
import librosa
from pathos.pools import ProcessPool as Pool
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import PyEMD
import vmdpy
from calcu_features import audio_augmentation

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class SignalDecomposition:
    """语音信号分解"""

    def __init__(self, audio_aug: bool = True, bgn_dir: str = ''):
        """
        初始化
        :return: None
        """
        self.audio_aug = audio_aug
        self.bgn_dir = bgn_dir
        self.audio_f_list = []  # 每个被试的所有音频文件组成的列表

    @staticmethod
    def get_decomposition(signal: np.ndarray, method: str = 'CEEMDAN',
                          show=False, point_scale=1.0, save_fig=None):
        """
        对信号进行类经验模态分解EMD
        :param signal: 输入信号
        :param method: 分解方法'EMD'/'CEEMDAN'/'VMD'
        :param show: 是否对分解后的信号分量（本征模态分量IMFs或乘积函数PF）进行显示绘制
        :param point_scale: 信号的横轴时间点尺度，即每个点对应的毫秒数
        :param save_fig: 图片保存路径，默认为None，不保存
        :return: 分解后的信号分量：np.ndarray[shape=(信号分量数, 采样点数), dtype=float64]
        """
        assert method in ['EMD', 'CEEMDAN', 'VMD'], "method仅接受'EMD'/'CEEMDAN'/'VMD'输入"
        signal = np.array(signal, dtype=np.float16)
        if method == 'CEEMDAN':
            emd_obj = PyEMD.EMD(DTYPE=np.float16, spline_kind='linear')
            # ceemdan = PyEMD.CEEMDAN(trials=20, ext_EMD=emd_obj, parallel=True,
            #                         processes=max(psutil.cpu_count(logical=False) // 5, 2))
            ceemdan = PyEMD.CEEMDAN(trials=20, ext_EMD=emd_obj)
            ceemdan.noise_seed(rs)
            with warnings.catch_warnings():  # 防止EMD.py:748: RuntimeWarning: divide by zero encountered in true_divide
                warnings.simplefilter('ignore', RuntimeWarning)
                c_imfs = ceemdan.ceemdan(signal, max_imf=10)
            c_residual = ceemdan.residue
            imfs_residual = np.vstack((c_imfs, c_residual))
        elif method == 'VMD':
            imfs_residual, u_hat, cen_fs = vmdpy.VMD(signal, alpha=2000, tau=0, K=10, DC=0,
                                                     init=1, tol=1e-6)
        else:  # EMD
            emd = PyEMD.EMD(DTYPE=np.float16, spline_kind='akima')
            with warnings.catch_warnings():  # 防止EMD.py:748: RuntimeWarning: divide by zero encountered in true_divide
                warnings.simplefilter('ignore', RuntimeWarning)
                imfs_residual = emd.emd(signal, max_imf=10)
        if show:
            mms = MinMaxScaler((-1, 1))
            mms_signal = mms.fit_transform(signal.reshape(-1, 1))
            imf_res_num = imfs_residual.shape[0]
            fig, ax = plt.subplots(imf_res_num + 1, 1, sharex='col', figsize=(7, 9))
            ax[0].plot([i * point_scale for i in range(len(signal))], mms_signal.reshape(len(signal), -1), 'r')
            ax[0].tick_params(labelsize=8)
            ax[0].set_ylabel("Signal", fontdict={'fontsize': 9})
            ax[0].set_ylim(-1.0, 1.0)
            ylabel = 'IMF '
            for i in range(imf_res_num):
                mms_imfs_residual = mms.transform(imfs_residual[i].reshape(-1, 1))
                ax[i + 1].plot([i * point_scale for i in range(len(imfs_residual[i]))],
                               mms_imfs_residual.reshape(len(imfs_residual[i]), -1), 'g', lw=1.0)
                ax[i + 1].tick_params(labelsize=8)
                if i < imf_res_num - 1:
                    ax[i + 1].set_ylabel(ylabel + str(i + 1), fontdict={'fontsize': 9})
                else:
                    ax[i + 1].set_ylabel("Residual", fontdict={'fontsize': 9})
                ax[i + 1].set_ylim(-1.0, 1.0)
            ax[-1].set_xlabel("Time (ms)", fontdict={'fontsize': 12})
            fig.align_ylabels()
            fig.text(0.03, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical', fontdict={'fontsize': 12})
            plt.tight_layout(rect=(0.05, 0, 1, 1))
            if save_fig is not None:
                plt.savefig(save_fig, dpi=600, bbox_inches='tight', pad_inches=0.2)
            # plt.show()
            plt.close()
        return imfs_residual

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
        group = os.path.normpath(audio_f).split(os.sep)[-4]
        subject_id = os.path.normpath(audio_f).split(os.sep)[-1].split("_")[0]
        save_path = os.path.join(res_dir, group, subject_id, method)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        if self.audio_aug:
            aud_f_data = audio_augmentation(audio_f, self.bgn_dir)
            for aud_f_d in aud_f_data:
                save_f = os.path.join(save_path, aud_f_d[0] + '.npy')
                if not os.path.exists(save_f):
                    imfs_residual = self.get_decomposition(aud_f_d[-1], method, show=False)
                    np.save(save_f, imfs_residual)
                    print(f'{subject_id}_{aud_f_d[0]} npy file saved ({method})')
                else:
                    print(f'{subject_id}_{aud_f_d[0]} npy file already exists, skipped ({method})')
        else:
            signal, sr = librosa.load(audio_f, sr=None)
            save_name = os.path.basename(audio_f).replace(".wav", "")
            save_f = os.path.join(save_path, save_name + '.npy')
            if not os.path.exists(save_f):
                imfs_residual = self.get_decomposition(signal, method, show=False)
                np.save(save_f, imfs_residual)
                print(f'{subject_id}_{save_name} npy file saved ({method})')
            else:
                print(f'{subject_id}_{save_name} npy file already exists, skipped ({method})')

    def run_parallel(self, data_dir: str = '', res_dir: str = '', method: str = 'CEEMDAN', n_jobs=None):
        """
        并行处理，保存所有信号分量至本地文件
        :param data_dir: 数据文件路径
        :param res_dir: 结果保存路径
        :param method: 分解方法'EMD'/'CEEMDAN'/'VMD'
        :param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
        :return: None
        """
        assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
        if n_jobs == -1:
            n_jobs = None
        for wav_file in sorted(glob.glob(os.path.join(data_dir, r"**/*.wav"), recursive=True)):
            self.audio_f_list.append(wav_file)
        _parallel_process = partial(self.decomposition_save, res_dir=res_dir, method=method)
        with Pool(n_jobs) as pool:
            pool.map(_parallel_process, self.audio_f_list)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(
        f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.getcwd()  # 获取当前文件夹
    # audio_data_prep = os.path.join(current_path, 'data/audio/WDHC/preprocess')
    audio_data_prep = DATA_PATH_PREP
    audio_data_bgn = os.path.join(current_path, 'data/audio/DomesticNoise')
    output_path_raw = os.path.join(current_path, r'data/decomposition/WDHC/preprocess')
    output_path_aug = os.path.join(current_path, r'data/decomposition/WDHC/augmentation')
    mds = {'CEEMDAN': 30, }

    for md in mds.keys():  # 获取EMD信号分量，一个1min音频分解需用时约6:37:56.894245（集合大小为20，线性插值，18核并行计算）
        print(f"---------- Begin {md} ----------")
        sd = SignalDecomposition(audio_aug=True, bgn_dir=audio_data_bgn)
        sd.run_parallel(audio_data_prep, output_path_aug, md, n_jobs=mds[md])

    end_time = datetime.datetime.now()
    print(f"---------- End Time ({os.path.basename(__file__)}): {end_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    print(f"---------- Time Used ({os.path.basename(__file__)}): {end_time - start_time} ----------")
    with open(r"./results/finished.txt", "w") as ff:
        ff.write(f"------------------ Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
        ff.write(f"------------------ Time Used {end_time - start_time} "
                 f"({os.path.basename(__file__)}) -------------------\r\n")
