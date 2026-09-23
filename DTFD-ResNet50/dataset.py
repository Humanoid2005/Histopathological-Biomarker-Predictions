import os
import random
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class Datapoint:
    def __init__(self,embedding_path,label,split=None):
        self.embedding_path = embedding_path
        self.label = label
        self.split = split

class BiomarkerDataset(Dataset):
    def __init__(self,csv_path,data_path,biomarker,split_info_path):
        self.csv_path = csv_path
        self.data_path = data_path
        self.split_info_path = split_info_path
        self.datapoints = []
        self.train_split = []
        self.val_split = []
        self.biomarker = biomarker

    def load_data(self):
        splits = self.load_split_info(path=self.split_info_path)

        if splits is None:
            df = pd.read_csv(self.csv_path)
            df['Case Number'] = df['Case Number'].astype(str)
            data_dir = Path(self.data_path)
            for file_path in data_dir.iterdir():
                if file_path.is_file():
                    filename_without_ext = file_path.stem 
                    matched_row = df[df['Case Number'] == filename_without_ext]
                    if not matched_row.empty:
                        label = 1 if matched_row.iloc[0][self.biomarker] == 1 else 0
                        new_datapoint = Datapoint(embedding_path=str(file_path), label=label)
                        self.datapoints.append(new_datapoint)
            self.train_split,self.val_split = self.split_data()
            self.save_split_info(self.train_split,self.val_split,path=self.split_info_path)
            self.datapoints = self.train_split + self.val_split
        else:
            self.train_split = [Datapoint(**d) for d in splits if d['split'] == 'train']
            self.val_split = [Datapoint(**d) for d in splits if d['split'] == 'val']
            self.datapoints = self.train_split + self.val_split

    def analyse(self,log=True):
        positive_datapoints = [dp for dp in self.datapoints if dp.label == 1]
        negative_datapoints = [dp for dp in self.datapoints if dp.label != 1]
        if log:
            print(f"Positive: {len(positive_datapoints)}, Negative: {len(negative_datapoints)}")
        return positive_datapoints,negative_datapoints

    def split_data(self):
        unique_dps = list({dp.embedding_path: dp for dp in self.datapoints}.values())
        pos_dps = [dp for dp in unique_dps if dp.label == 1]
        neg_dps = [dp for dp in unique_dps if dp.label != 1]
        
        random.shuffle(pos_dps)
        random.shuffle(neg_dps)
        
        pos_split_idx = int(0.8 * len(pos_dps))
        neg_split_idx = int(0.8 * len(neg_dps))
        
        val_pos = pos_dps[pos_split_idx:]
        val_neg = neg_dps[neg_split_idx:]
        val_datapoints = val_pos + val_neg
        random.shuffle(val_datapoints)
        
        train_pos = pos_dps[:pos_split_idx]
        train_neg = neg_dps[:neg_split_idx]
        
        target_size = max(len(train_pos), len(train_neg))
        
        if len(train_pos) < target_size:
            train_pos.extend(random.choices(train_pos, k=target_size - len(train_pos)))
        elif len(train_neg) < target_size:
            train_neg.extend(random.choices(train_neg, k=target_size - len(train_neg)))
            
        train_datapoints = train_pos + train_neg
        random.shuffle(train_datapoints)

        for dp in train_datapoints:
            dp.split = 'train'

        for dp in val_datapoints:
            dp.split = 'val'
        
        return train_datapoints, val_datapoints
        
    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self,split,idx):
        if split == 'train':
            datapoint = self.train_split[idx]
        elif split == 'val':
            datapoint = self.val_split[idx]
            
        data = torch.load(datapoint.embedding_path)
        label = torch.tensor(datapoint.label, dtype=torch.float32)
        
        return data, label

    def save_split_info(self,train_split,val_split,path):
        data = [dp.__dict__ for dp in train_split + val_split]
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df.to_csv(path,index=False)

    def load_split_info(self,path):
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient='records')
        except FileNotFoundError:
            return None

class SplitWrapper(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
        
    def __len__(self):
        if self.split == 'train':
            return len(self.dataset.train_split)
        elif self.split == 'val':
            return len(self.dataset.val_split)
        return len(self.dataset.datapoints)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.split, idx)

class BiomarkerDataLoader(DataLoader):
    def __init__(self, dataset, split, batch_size, **kwargs):
        wrapped_dataset = SplitWrapper(dataset, split)
        shuffle = kwargs.pop('shuffle', split == 'train')
        super().__init__(wrapped_dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

