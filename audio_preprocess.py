# -*- coding: utf-8 -*-
# Copyright (c) 2022. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time     : 2022/8/23 9:06
# @Author   : ZL.Z
# @Email    : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : audio_preprocess.py
# @Software : Python3.9; PyCharm; Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-79-generic x86_64)
# @Hardware : 2*X640-G30(XEON 6258R 2.7G); 3*NVIDIA GeForce RTX3090 (24G)
# @Version  : V2.1: 2023/3/3
#             修改音频预处理过程：包括格式转换、裁剪、降噪、归一化
#             V2.0: 2023/1/30 - 2023/1/31
#             1. 添加结构化自发言语任务实验；
#             2. 重新匹配数据，保证年龄T检验不显著
#             V1.0: 2022/8/23
#             First version.
# @License  : None
# @Brief    : 自发言语音频预处理

from config import *
import subprocess
import glob
import csv
import shutil
import librosa
import soundfile as sf
import parselmouth
import datetime
from pathos.pools import ProcessPool as Pool
import noisereduce as nr
from typing import Optional


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


def audio_preprocess(audio_file: str, output_folder: str, from_time: Optional[float] = None,
                     to_time: Optional[float] = None, normalize=True, denoise=False):
    """
    音频预处理：包括格式转换、裁剪、降噪、归一化
    :param audio_file: 待处理音频文件
    :param output_folder: 输出文件夹
    :param from_time: 截断音频，音频开始时间，默认不截断
    :param to_time: 截断音频，音频结束时间，默认不截断
    :param normalize: 是否进行归一化，幅值归一化至[-1,1]，默认是
    :param denoise: 是否进行降噪，由于降噪对于不同音频效果有差异，默认否
    :return: None
    """
    print(audio_file + "音频数据处理中...")
    if not os.path.exists(audio_file):
        raise FileExistsError(audio_file + "输入文件不存在！")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    audio_name = os.path.basename(audio_file)
    output_audio = os.path.join(output_folder, audio_name)
    temp_audio = os.path.join(output_folder, "temp_" + audio_name)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    print("----------{} STEP1: 音频格式转换----------".format(audio_name))
    # 调用ffmpeg，将任意格式音频文件转换为.wav文件，pcm有符号16bit,1：单通道,16kHz，不显示打印信息
    subprocess.run("ffmpeg -loglevel quiet -y -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s" %
                   (audio_file, temp_audio), shell=True)
    wav, sr = librosa.load(temp_audio, sr=None)
    if denoise:
        print("----------{} STEP2: 降噪----------".format(audio_name))
        reduced_noise = nr.reduce_noise(y=wav, sr=sr, n_fft=512, prop_decrease=0.9)
    else:
        reduced_noise = wav
    if normalize:
        print("----------{} STEP3: 归一化----------".format(audio_name))
        wav_norm = librosa.util.normalize(reduced_noise)
    else:
        wav_norm = reduced_noise
    sf.write(temp_audio, wav_norm, sr, subtype="PCM_16")
    sound = parselmouth.Sound(temp_audio).extract_part(from_time, to_time,
                                                       parselmouth.WindowShape.RECTANGULAR, 1.0, False)
    sound.save(output_audio, parselmouth.SoundFileFormat.WAV)
    if os.path.exists(temp_audio):
        os.remove(temp_audio)


def run_audio_preprocess_parallel(original_path: str, preprocessed_path: str, n_jobs=None, **kwargs):
    """
    并行运行音频预处理
    :param original_path: 原始数据文件路径
    :param preprocessed_path: 预处理保存数据文件路径
    :param n_jobs: 并行运行CPU核数，默认为None，取os.cpu_count()全部核数,-1/正整数/None类型
    :param kwargs: audio_preprocess参数
    :return: None
    """
    assert (n_jobs is None) or (type(n_jobs) is int and n_jobs > 0) or (n_jobs == -1), 'n_jobs仅接受-1/正整数/None类型输入'
    if n_jobs == -1:
        n_jobs = None
    for each_file in os.listdir(original_path):
        if each_file == "HC" or each_file == "WD":
            preprocessed_p = os.path.join(preprocessed_path, each_file)
            if not os.path.exists(preprocessed_p):
                os.makedirs(preprocessed_p)
            data_path = os.path.join(original_path, each_file)
            for root, dirs, files in os.walk(data_path):

                def parallel_process(name):
                    if name.split('.')[0].split('_')[0] not in id_used:
                        return
                    if name.endswith('.csv'):  # 将csv文件复制到目标文件夹下
                        csv_file = os.path.join(root, name)
                        dst_csv_p = root.replace(os.path.abspath(original_path), os.path.abspath(preprocessed_path))
                        dst_csv_p = dst_csv_p.replace("/female/", "/").replace("/male/", "/").replace("session_1", "")
                        if 'session_' not in dst_csv_p:
                            if not os.path.exists(dst_csv_p):
                                os.makedirs(dst_csv_p)
                            shutil.copy(csv_file, os.path.join(dst_csv_p, name))
                    if name.endswith('.wav'):  # 遍历处理.wav文件
                        wav_file = os.path.join(root, name)
                        if 'session_1' in wav_file:
                            if '12_SI' in wav_file:
                                output_path = root.replace(os.path.abspath(original_path),
                                                           os.path.abspath(preprocessed_path)).split('session_1')[0] + 'SI'
                            elif '10_PD/CookieTheft' in wav_file:
                                output_path = root.replace(os.path.abspath(original_path),
                                                           os.path.abspath(preprocessed_path)).split('session_1')[0] + 'PD'
                            else:
                                return
                        else:
                            if '14_SI' in wav_file:
                                output_path = root.replace(os.path.abspath(original_path),
                                                           os.path.abspath(preprocessed_path)).split('14_SI')[0] + 'SI'
                            elif '12_PD/CookieTheft' in wav_file:
                                output_path = root.replace(os.path.abspath(original_path),
                                                           os.path.abspath(preprocessed_path)).split('12_PD')[0] + 'PD'
                            else:
                                return
                        output_path = output_path.replace("/female/", "/").replace("/male/", "/")
                        audio_preprocess(wav_file, output_path, **kwargs)

                # 使用设定数量的CPU核数（这里有闭包，不可pickle，因此不能用multiprocessing中的Pool，这里用pathos）
                with Pool(n_jobs) as pool:
                    pool.map(parallel_process, files)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"---------- Start Time ({os.path.basename(__file__)}): {start_time.strftime('%Y-%m-%d %H:%M:%S')} ----------")
    current_path = os.path.dirname(os.path.realpath(__file__))
    # preprocessed_data = DATA_PATH_PREP  # 示意，为了省空间，预处理后的数据存在DATA_PATH_PREP中，留为共用
    preprocessed_data = os.path.join(current_path, "data/audio/WDHC/preprocess")
    if os.path.exists(preprocessed_data):
        shutil.rmtree(preprocessed_data)
    for original_data in [DATA_PATH, DATA_PATH_EXT1, DATA_PATH_EXT2]:
        run_audio_preprocess_parallel(original_data, preprocessed_data, n_jobs=-1,
                                      from_time=0.0, to_time=60.0, normalize=True, denoise=True)

    subid, nam, sex, age, group = [], [], [], [], []
    for i_f in glob.iglob(os.path.join(preprocessed_data, r"**/*.csv"), recursive=True):
        if int(os.path.basename(i_f)[:6]) >= 202209:
            csv_data = pd.read_csv(i_f)
            nam.append(csv_data['name'][0])
            sex.append({"男": 1, "女": 0}[csv_data['sex'][0]])
            age.append(int(csv_data['age'][0]))
        else:
            csv_data = read_csv(i_f)
            nam.append(csv_data[0][0].split("：")[1])
            if int(os.path.basename(i_f)[:6]) <= 202112:
                sex.append({"男": 1, "女": 0}[csv_data[0][3].split("：")[1]])
            else:
                sex.append({"男": 1, "女": 0}[csv_data[0][2].split("：")[1]])
            age.append(int(csv_data[0][1].split("：")[1]))
        subid.append(os.path.basename(i_f).rstrip('.csv'))
        group.append(os.path.normpath(i_f).split(os.sep)[-3])
    subinfo = pd.DataFrame({"id": subid, "name": nam, "sex": sex, "age": age, "group": group})
    subinfo.sort_values(by=['group', 'sex', 'id'], inplace=True)
    subinfo.to_csv(os.path.join(current_path, 'data/subinfo.csv'), index=False, encoding='utf-8-sig')

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

