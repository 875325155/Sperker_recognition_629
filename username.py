# -*- coding: utf-8 -*-
"""
@author: GENESIS
"""


import numpy as np
import scipy
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn.mixture import GMM
import pickle
import os
import mfeatures
import warnings
warnings.filterwarnings("ignore")
import pyaudio
import wave
import time
def record():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 10
    WAVE_OUTPUT_FILENAME = "testfile.wav" 
    INPUT_DEVICE_INDEX=1
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,input_device_index=INPUT_DEVICE_INDEX)
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def test1():

    #source   = "testfile.wav"
    modelpath = "models\\"

    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    # print(gmm_files)

    #Load the Gaussian gender Models
    #pickle.load(file)
    #反序列化对象，将文件中的数据解析为一个python对象。file中有read()接口和readline()接口
    #得到模型
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname
                  in gmm_files]
    # Read the test directory and get the list of test audio files
    print("Do you want to Test a Single Audio: Press '1' or The complete Test Audio Sample: Press '0' ?")
    take = int(input().strip())

    # 触发异常后，后面的代码就不会再执行
    if (take != 0 and take != 1):
        raise ValueError("请输入0或1！")
    if take == 1:
        path="testfile.wav"
        #rate,sig = wav.read(source+path)
        '''Returns
        rate : int
            Sample rate of wav file.
        data : numpy array
            Data read from wav file.  Data-type is determined from the file;
            see Notes.
        '''
        rate,sig = wav.read(path)
        #得到mfcc
        mfcc_feat=mfeatures.extract_features(sig,rate)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm    = models[i]  #checking with each model one by one
            scores = np.array(gmm.score(mfcc_feat))
            # print(len(scores))
            log_likelihood[i] = scores.sum()
            print(speakers[i]+str(log_likelihood[i]))
        winner = np.argmax(log_likelihood)
        z=speakers[winner]
        length=len(z)
        i=0
        while i<length:
         y=z[i]
         if y.isdigit():
            break
         i+=1
        result=z[0:i]
        result=result.upper()
        print("Detected as - ", result)
        return result

    elif take == 0:
        total_sample=0.0
        error=0
        test_file = "testSamplePath.txt"
        file_paths = open(test_file, 'r')

        # Read the test directory and get the list of test audio files
        for path in file_paths:

            total_sample += 1.0
            path = path.strip()
            print("Testing Audio : ", path)
            sr, audio = wav.read( path)
            vector = mfeatures.extract_features(audio, sr)

            log_likelihood = np.zeros(len(models))

            for i in range(len(models)):
                gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

            winner = np.argmax(log_likelihood)
            print("Detected as - ", speakers[winner])
            checker_name = path.split("_")[0]
            if speakers[winner] != checker_name:
                error += 1
            time.sleep(1.0)

        print(error, total_sample)
        accuracy = ((total_sample - error) / total_sample) * 100

        print("The Accuracy Percentage for the current testing Performance with MFCC + GMM is : ", accuracy, "%")


# test1()
