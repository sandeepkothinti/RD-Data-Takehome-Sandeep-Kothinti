import os
import pandas as pd
import random
import shutil
from sklearn.model_selection import train_test_split

# setup paths
data_path_original = '../real_and_fake_face/'
real_path = data_path_original+'/training_real/'
real_path_copy = '../data/'
fake_path = data_path_original+'/training_fake/'

# dataframe to store data
df = pd.DataFrame({'filename':[], 'filepath':[], 'label':[], 'difficulty':[], 'bit_mask':[], 'train_split':[]})

# sampling real data

data_real = os.listdir(real_path)
real_samples_train = random.sample(data_real, 270)
real_train, real_rem = train_test_split(real_samples_train, test_size=0.33)
real_valid, real_test = train_test_split(real_rem, test_size=0.5)
split_labels = ['train', 'valid', 'test']
for idx, split in enumerate((real_train, real_valid, real_test)):
    for im in split:
        shutil.copyfile(real_path+im, real_path_copy+im)
        df.loc[len(df.index)]=[im, real_path_copy+im, 'real', 'N/A', 'N/A', split_labels[idx]]

# sampling fake data
df_tmp = pd.DataFrame({'filename':[], 'filepath':[], 'label':[], 'difficulty':[], 'bit_mask':[], 'train_split':[]})
data_fake = os.listdir(fake_path)
for im in data_fake:
    if im.startswith('.'):
        continue
    im_difficulty = im.split('_')[0]
    im_mask = im[:-4].split('_')[-1]
    df_tmp.loc[len(df_tmp.index)] = [im, real_path_copy+im, 'fake', im_difficulty, im_mask, 'N/A']

masks = set(df_tmp['bit_mask'].to_list())
split_labels = ['train', 'train','train','train', 'valid', 'test', 'test', 'valid', 'train', 'train', 'train']
missing=0
for diff in ['easy', 'mid', 'hard']:
    for mask in masks:
        df_select = df_tmp.loc[(df_tmp.difficulty==diff)&(df_tmp.bit_mask==mask)]
        if len(df_select) >6+missing:
            df_add = df_select.sample(6+missing)
            missing=0
        else:
            df_add = df_select
            missing = 6- len(df_add)
        for j in range(len(df_add)):
            shutil.copyfile(fake_path+df_add.iloc[j].filename, real_path_copy+df_add.iloc[j].filename)
            df_add.iloc[j].train_split = split_labels[j]
        df=df.append(df_add, ignore_index=True)
print(len(df.loc[(df.label=='fake')&(df.train_split=='test')]))
df.to_csv('full_data.csv')
