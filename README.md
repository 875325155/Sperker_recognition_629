文件介绍：

1. generate_voxforge_training_data  ->从voxforge之中读取MFCC数据，并将其转为pkl
2. generate_voxforge_txt_files-> 每个示例中，Voxforge下载文件中一个文件中包含的提示将转换为一个txt文件，稍后将用于生成预测目标。
3. GMM1-> 模型的训练
4. username.py  ->  用来从麦克风输入音频数据和对数据进行测试，提供两个接口，可以对标准语音库进行测试，亦可通过对用户自己输入的数据进行测试
5. trainrecord.py ->用来从麦克风输入用户的数据
6. train.py ->定义训练的界面
7. test.py ->定义测试的界面
8.  mfeatures.py->提取mfcc参数
9. db.py ->定义测试结果的界面，在测试界面之后出现
10. homeapp.py ->主界面模块
11. models->说话者的语音模型（gmm）
12. Models->语料库里的语音模型(gmm)

实验方面的思路

1. GMM1 45行count的取值 (1)可以做实验找较好的值 (2)可以让用户自己输入
2. 



你说你 想要逃

偏偏注定要落脚

情灭了 爱熄了

剩下空心要不要

春已走 花又落

用心良苦却成空

