import numpy as np
import scipy
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.mixture import GMM

import pickle
import os
import warnings
import MFCC_features

warnings.filterwarnings("ignore")


# 采用Log Filterbank Energies为参数


# 训练用户的数据
def traine(x):
    file_paths = open("development_set_enroll.txt", 'w')
    i = 1
    while i < 6:
        file_paths.write(x + str(i) + ".wav\n")
        i += 1
    file_paths.close()

    # path to training data
    source = r"training_data/"
    # path where training speakers will be saved
    dest = r"models/"
    train_file = "development_set_enroll.txt"
    file_paths = open(train_file, 'r')

    count = 1
    # Extracting features for each speaker (5 files per speakers)
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        # read the audio
        rate, sig = wav.read(source + path)
        mfcc_feat = MFCC_features.extract_features(sig, rate)
        # extract MFCC

        if features.size == 0:
            features = mfcc_feat
        else:
            features = np.vstack((features, mfcc_feat))
        # when features of 5 files of speaker are concatenated, then do model training
        # 这里可以修改为更多的count，可以为一个实验的点
        if count == 5:
            gmm = GMM(n_components=16, n_iter=200, covariance_type='diag', n_init=10)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1


# **********************************************
# 得到所有语音文件的完整路径
voxforge_data_dir = './Voxforge'


def list_files_for_speaker(folder):
    """
    Generates a list of wav files for a given speaker from the voxforge dataset.
    Args:
        speaker: substring contained in the speaker's folder name, e.g. 'Aaron'
        folder: base folder containing the downloaded voxforge data

    Returns: list of paths to the wavfiles
    """
    train_file = "train_Voxforge_enroll.txt"
    test_file = "test_Voxforge_enroll.txt"
    name_file = "Voxforge_model.txt"
    train_paths = open(train_file, 'w')
    test_paths = open(test_file, 'w')
    name_paths = open(name_file, 'w')
    speaker_folders = [d for d in os.listdir(folder)]
    # print(speaker_folders)
    wav_files = []
    # d为子目录
    for d in speaker_folders:
        '''
        此处需要列出文件下所有wav的总数，并且将其5个放入测试txt，5个放入训练txt
        利用循环遍历来实现
        '''
        # 得到文件夹的wav文件数
        # print(d)
        if d.split("-")[0] not in wav_files:
            wav_files.append(d.split("-")[0])
            a = len(os.listdir(os.path.join(folder, d, 'wav')))
            count = 1
            if a == 10:
                name_paths.write(d + '\n')
                for f in os.listdir(os.path.join(folder, d, 'wav')):
                    # print(f)
                    if (count <= 5):
                        train_paths.write(os.path.join(d, 'wav', f) + '\n')
                    # wav_files.append(os.path.abspath(os.path.join(folder, d, 'wav', f)))
                    else:
                        test_paths.write(os.path.join(d, 'wav', f) + '\n')
                    count += 1

    # return wav_files
    test_paths.close()
    train_paths.close()
    name_paths.close()


# 训练voxforge语料库之中的数据
def train_voxforge():
    # file_paths = open("Voxforge_model.txt", 'w')
    # i = 1
    # while i < 6:
    #     file_paths.write(x + str(i) + ".wav\n")
    #     i += 1
    # file_paths.close()
    # path to training data
    source = r"Voxforge/"
    # path where training speakers will be saved
    dest = r"Voxforge_models/model_3_lfe/"
    if not os.path.exists(dest):  # 如果路径不存在
        os.makedirs(dest)
    train_file = "train_Voxforge_enroll.txt"
    file_paths = open(train_file, 'r')

    count = 1
    # Extracting features for each speaker (5 files per speakers)
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        # read the audio
        # print(source + path)
        rate, sig = wav.read(source + path)
        mfcc_feat = MFCC_features.extract_features(sig, rate)
        # extract MFCC

        if features.size == 0:
            features = mfcc_feat
        else:
            features = np.vstack((features, mfcc_feat))
        # when features of 5 files of speaker are concatenated, then do model training
        # 这里可以修改为更多的count，可以为一个实验的点
        if count == 5:
            print(path.split("-")[0] + 'is training!')
            gmm = GMM(n_components=16, n_iter=200, covariance_type='diag', n_init=10)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1


# 得到各个需要的txt
# list_files_for_speaker(voxforge_data_dir)
# 对voxforge数据进行训练，并进行序列化，将结果保存为.gmm文件
train_voxforge()

















