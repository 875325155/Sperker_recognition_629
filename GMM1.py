import numpy as np
import scipy
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.mixture import GMM

import pickle
import os
import warnings
import mfeatures
warnings.filterwarnings("ignore")


def traine(x):
    file_paths=open("development_set_enroll.txt" ,'w')
    i=1
    while i<6:
        file_paths.write(x+str(i)+".wav\n")
        i+=1
    file_paths.close()
     
    #path to training data
    source   = r"training_data/"
    #path where training speakers will be saved
    dest = r"models/"
    train_file = "development_set_enroll.txt"        
    file_paths = open(train_file,'r')
    
    count = 1
    # Extracting features for each speaker (5 files per speakers)
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()
        # read the audio
        rate,sig = wav.read(source+path)
        mfcc_feat=mfeatures.extract_features(sig,rate)
        # extract MFCC 
        
        if features.size == 0:
            features = mfcc_feat
        else:
            features = np.vstack((features, mfcc_feat))
        # when features of 5 files of speaker are concatenated, then do model training
        #这里可以修改为更多的count，可以为一个实验的点
        if count == 5:
            gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 10)
            gmm.fit(features)
            
            # dumping the trained gaussian model
            picklefile = path.split("-")[0]+".gmm"
            pickle.dump(gmm,open(dest + picklefile,'wb'))
            features = np.asarray(())
            count = 0
        count = count + 1

#**********************************************
#所有语音文件的完整路径
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

    train_paths = open(train_file, 'w')
    test_paths  = open(test_file, 'w')
    speaker_folders = [d for d in os.listdir(folder)]
    # print(speaker_folders)
    # wav_files = []
    #d为子目录
    for d in speaker_folders:
        '''
        此处需要列出文件下所有wav的总数，并且将其5个放入测试txt，5个放入训练txt
        利用循环遍历来实现
        '''
        #得到文件夹的wav文件数
        a=len(os.listdir(os.path.join(folder, d, 'wav')))
        count=1
        if a==10:
            for f in os.listdir(os.path.join(folder, d, 'wav')):
                # print(f)
                if(count<=5):
                    train_paths.write(os.path.join(d, 'wav', f)+'\n')
                # wav_files.append(os.path.abspath(os.path.join(folder, d, 'wav', f)))
                else:
                    test_paths.write(os.path.join(d, 'wav', f) + '\n')
                count+=1


    # return wav_files
    test_paths.close()
    train_paths.close()
#训练voxforge语料库之中的数据
def train_voxforge(x):
    file_paths = open("Voxforge_model.txt", 'w')

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
        mfcc_feat = mfeatures.extract_features(sig, rate)
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

list_files_for_speaker(voxforge_data_dir)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
