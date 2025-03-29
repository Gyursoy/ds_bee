import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from ..utils.config import config


TRAIN = 'train'
VAL = 'val'
TEST = 'test'


class AudioDataset(Dataset):

    stat_features = ['zcr', 'rms', 'sce', 'lpc']
    img_features = ['mel', 'mfc', 'stf', 'chr', 'dmf', 'psd']

    def __init__(self, df, transform=None, selected_features=None):
        
        # df = pd.read_csv(Path(config['data']['df_model_path']))

        # if config['model']['prediction_type'] == 'regression':
        #     self.df = self.construct_regression_df(df)
        # else:
        #     self.df = df
        self.df = df
            
        self.data_folder = Path(config['data']['data_folder'])
        self.dir_dict = config['data']['dir_dict']
        self.transform = transform
        self.selected_features = selected_features
        self.pred_type = config['model']['prediction_type']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.pred_type == 'classification':
            label = self.df.iloc[idx]['foulbrood']
        else:
            label = self.df.iloc[idx]['timedelta_hours']

        label = torch.tensor(label)

        imgs = {}
        for feature, dir_name in self.dir_dict.items():
            if feature in self.stat_features:
                feat_name = dir_name + self.df.iloc[idx]['audio'][:-4] + ".npz"
                imgs[feature] = np.load(self.data_folder / dir_name / feat_name)['arr_0']
                imgs[feature] = torch.tensor(imgs[feature])
                imgs[feature] = imgs[feature].unsqueeze(0)
            else:     
                img_name = dir_name + self.df.iloc[idx]['audio'][:-4] + ".jpg"
                imgs[feature] = read_image(str(self.data_folder / dir_name / img_name))
            
                if self.transform:
                    imgs[feature] = self.transform(imgs[feature])

            if self.selected_features is not None:
                imgs[feature] = imgs[feature][:, self.selected_features[feature]]

        return imgs, label
    

    



    
def construct_regression_df(df):

    with open('Data\-defect_files.json', "r") as f:
        defect_files = json.load(f)
    
    df_new = df[np.array([0 if i in defect_files else 1 for i in df['audio']], dtype = bool)]
    df_new = df_new[df_new['hive_id'] == 22]
    df_new.reset_index(drop=True, inplace=True)
    df_new['date'] = pd.to_datetime(df_new['date'], format='%y%m%d')
    df_new['time'] = df_new['time'].apply(lambda x: f'{int(x):06d}')
    df_new['datetime'] = pd.to_datetime(df_new['date'].dt.strftime('%Y-%m-%d') + ' ' + df_new['time'], format='%Y-%m-%d %H%M%S')
    df_new.drop(columns=['date', 'time', 'Unnamed: 0'], inplace=True)
    df_new.sort_values(['datetime'], inplace=True)
    last_datetime = df_new['datetime'].max()
    df_new['timedelta'] = last_datetime - df_new['datetime']
    df_new['timedelta_hours'] = df_new['timedelta'].dt.total_seconds() / (60 * 60)
    df_new['timedelta_hours'] = df_new['timedelta_hours'].astype(float)

    df_model = df_new[df_new['timedelta_hours'] < 240]

    
    return df_model



def split_data(df, seed=42):
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=seed)

    return df_train, df_test, df_val

def get_dataframes(df):
    df_splits = split_data(df)
    processed_df_splits = dict(zip([TRAIN, TEST, VAL],
                               list(split_data(df))))

    return processed_df_splits