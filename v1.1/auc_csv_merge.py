#尝试直接读取文件夹内所有csv，记得看看列表，是不是读对了
import glob
import pandas as pd
import numpy as np

io = glob.glob(r"*.csv")
len_io=len(io)
print('总共输入表的数量为：',len_io)
prob_list=[]

for i in range(len_io):
    sub_1 =  pd.read_csv(io[i])
    denominator=len(sub_1)
    for my_classes in ['healthy','multiple_diseases','rust','scab']:
        sub_label_1 = sub_1.loc[:, my_classes].values
        sort_1=np.argsort(sub_label_1)
        for i,temp_sort in enumerate(sort_1):
            sub_label_1[temp_sort]=i/denominator
            sub_1.loc[:,my_classes]=sub_label_1
    prob_list.append(sub_1.loc[:,'healthy':].values)

sub_1.loc[:,'healthy':] = np.mean(prob_list,axis =0)
sub_1.to_csv('out/submission.csv', index=False)
print(sub_1.head())