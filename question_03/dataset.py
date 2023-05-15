from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class myDS(Dataset):
    def __init__(self, csv_file, mode='train', out_size=[64,64]):
        self.df_total = pd.read_csv(csv_file)
        self.out_size = out_size
        self.df_select = self.df_total.loc[(self.df_total.train_split==mode)]

    def __len__(self):
        return len(self.df_select)
    
    def __getitem__(self, index):
        filepath = self.df_select.iloc[index].filepath
        label = np.asarray(self.df_select.iloc[index].label=='fake', dtype=np.float32)
        image = Image.open(filepath)
        image = np.asarray(image.convert('L').resize(self.out_size), dtype=np.float32)
        image = (image-np.min(image))/(np.max(image)-np.min(image))
        return image, label
        

