import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from tqdm import tqdm
import torch_geometric.transforms as T


class HeteroTrainData(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(HeteroTrainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['DoubleBox_Coupler_Train.dataset']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        
        df = pd.read_csv("DoubleBox_Coupler/DoubleBox_train.csv")
        unique_Seq = pd.DataFrame(data={"Seq":pd.RangeIndex(len(df["seq"]))})
        df["Seq"] = unique_Seq
        edge_len = 4

        grouped = df.groupby("Seq")
        for Seq, group in tqdm(grouped):
            
            data = HeteroData()
            
            data['ML'].x = torch.tensor(([[group.w1.values[0],edge_len], [group.w2.values[0],group.l2.values[0]], [group.w8.values[0],group.l3.values[0]], [group.w9.values[0],edge_len], 
                       [group.w3.values[0],group.l1.values[0]], [group.w7.values[0],group.l1.values[0]], [group.w10.values[0],group.l1.values[0]], 
                       [group.w4.values[0],edge_len], [group.w5.values[0],group.l2.values[0]], [group.w6.values[0],group.l2.values[0]], [group.w11.values[0],edge_len]]), 
                     dtype=torch.float)

            data['MT'].x = torch.tensor(([[group.w1.values[0],group.w2.values[0],group.w3.values[0]],[group.w2.values[0],group.w8.values[0],group.w7.values[0]],[group.w8.values[0],group.w9.values[0],group.w10.values[0]],
                                         [group.w5.values[0],group.w4.values[0],group.w3.values[0]],[group.w6.values[0],group.w5.values[0],group.w7.values[0]],[group.w11.values[0],group.w6.values[0],group.w10.values[0]]]), dtype=torch.float)
            data['ML'].y = torch.tensor(group.iloc[:,11:51].values[0], dtype=torch.float)
            data['MT', 'to', 'ML'].edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                                       [0, 1, 4, 1, 2, 5, 2, 3, 6, 7, 8, 4, 8, 9, 5, 9, 10, 6]], dtype=torch.long)
 
            data = T.ToUndirected()(data)
            data_list.append(data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        

        
class HeteroTestData(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(HeteroTestData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['DoubleBox_Coupler_Test.dataset']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        
        df = pd.read_csv("DoubleBox_Coupler/DoubleBox_test.csv")
        unique_Seq = pd.DataFrame(data={"Seq":pd.RangeIndex(len(df["seq"]))})
        df["Seq"] = unique_Seq
        edge_len = 4

        grouped = df.groupby("Seq")
        for Seq, group in tqdm(grouped):
            
            data = HeteroData()
            
            data['ML'].x = torch.tensor(([[group.w1.values[0],edge_len], [group.w2.values[0],group.l2.values[0]], [group.w8.values[0],group.l3.values[0]], [group.w9.values[0],edge_len], 
                       [group.w3.values[0],group.l1.values[0]], [group.w7.values[0],group.l1.values[0]], [group.w10.values[0],group.l1.values[0]], 
                       [group.w4.values[0],edge_len], [group.w5.values[0],group.l2.values[0]], [group.w6.values[0],group.l2.values[0]], [group.w11.values[0],edge_len]]), 
                     dtype=torch.float)

            data['MT'].x = torch.tensor(([[group.w1.values[0],group.w2.values[0],group.w3.values[0]],[group.w2.values[0],group.w8.values[0],group.w7.values[0]],[group.w8.values[0],group.w9.values[0],group.w10.values[0]],
                                         [group.w5.values[0],group.w4.values[0],group.w3.values[0]],[group.w6.values[0],group.w5.values[0],group.w7.values[0]],[group.w11.values[0],group.w6.values[0],group.w10.values[0]]]), dtype=torch.float)
            data['ML'].y = torch.tensor(group.iloc[:,11:51].values[0], dtype=torch.float)
            data['MT', 'to', 'ML'].edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
                                       [0, 1, 4, 1, 2, 5, 2, 3, 6, 7, 8, 4, 8, 9, 5, 9, 10, 6]], dtype=torch.long)
 
            data = T.ToUndirected()(data)
            data_list.append(data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])