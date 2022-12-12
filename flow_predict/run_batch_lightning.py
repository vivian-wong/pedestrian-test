import pandas as pd
import numpy as np 
import os
from scipy import interpolate
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.saving import load_hparams_from_yaml

import networkx as nx
import torch_geometric
from torch_geometric.utils import dense_to_sparse

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, WindmillOutputSmallDatasetLoader, WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal

from tqdm import tqdm


# # Dataset & Dataloaders

# In[4]:


def image_to_world(p, Homog):
    '''
    p: a np array of [x1, y1; x2, y2; ... ] 
    '''
    # from Opentraj repo
    pp = np.stack((p[:, 0], p[:, 1], np.ones(len(p))), axis=1) # append a rightmost column of 1s
    PP = np.matmul(Homog, pp.T).T # [world x, world y, 1] (3 x 3)
    P_normal = PP / np.repeat(PP[:, 2].reshape((-1, 1)), 3, axis=1)  # normalize since the third column is not exactly 1??
    return P_normal[:, :2]*0.8 # not sure why *0.8 but that's in the code. 

# parse original annotations to see which agents have been split due to noncontinuous timestamps
# note some trajs too short removed
def parse_gcs(path):
    # modified from https://github.com/crowdbotp/OpenTraj/blob/master/opentraj/toolkit/loaders/loader_gcs.py
    HOMOG = [[4.97412897e-02, -4.24730883e-02, 7.25543911e+01],
             [1.45017874e-01, -3.35678711e-03, 7.97920970e+00],
             [1.36068797e-03, -4.98339188e-05, 1.00000000e+00]]
    FPS = 25
    FRAME_STEPSIZE = 20
    
    raw_data_list = []  # the data to be converted into Pandas DataFrame
    tempdf = pd.DataFrame()
    file_list = sorted(os.listdir(path))
    
    for annot_file in file_list:
        annot_file_full_path = os.path.join(path, annot_file)
        with open(annot_file_full_path, 'r') as f:
            annot_contents = f.read().split()

        agent_id = int(annot_file.replace('.txt', ''))

        for i in range(len(annot_contents) // 3):
            py = float(annot_contents[3 * i])
            px = float(annot_contents[3 * i + 1])
            frame_id = int(annot_contents[3 * i + 2])
            raw_data_list.append([frame_id, agent_id, px, py])
    
    # fill frame_id, agent_id, x, y, timestamp
    df_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
    raw_data_frame = pd.DataFrame(np.stack(raw_data_list), columns=df_columns)
    raw_df_groupby = raw_data_frame.groupby("agent_id")
    trajs = [g for _, g in raw_df_groupby]
    
    # interpolate x,y from F=frame number, so that we can fill in xy for non-continuous timestamps
    for ii, tr in tqdm(enumerate(trajs)):
        if len(tr) < 2: continue
        interp_F = np.arange(tr["frame_id"].iloc[0], tr["frame_id"].iloc[-1], FRAME_STEPSIZE).astype(int)
        interp_func = interpolate.interp1d(tr["frame_id"], tr["pos_x"], kind='linear')
        interp_X_ = interp_func(interp_F)
        interp_func = interpolate.interp1d(tr["frame_id"], tr["pos_y"], kind='linear')
        interp_Y_ = interp_func(interp_F)
        agent_id = int(tr["agent_id"].iloc[0])
        tempdf = tempdf.append(pd.DataFrame({"frame_id": interp_F,
                                                       "agent_id": agent_id,
                                                       "pos_x": interp_X_,
                                                       "pos_y": interp_Y_}))
    raw_data_frame = tempdf.reset_index()
    
    # world coordinate
    world_coords = image_to_world(raw_data_frame[["pos_x", "pos_y"]].to_numpy(), HOMOG)
    raw_data_frame[["pos_x", "pos_y"]] = pd.DataFrame(world_coords)
    raw_data_frame["timestamp"] = raw_data_frame["frame_id"] / FPS
    
    raw_df_groupby = raw_data_frame.groupby("agent_id")
    
    # remove the trajectories shorter than 2 frames
    single_length_inds = raw_df_groupby.head(1).index[raw_df_groupby.size() < 2]
    raw_data_frame = raw_data_frame.drop(single_length_inds)
    raw_df_groupby = raw_data_frame.groupby("agent_id")
        
    # fill velocities
    dt = raw_df_groupby["timestamp"].diff()

    if (dt > (FRAME_STEPSIZE/FPS+0.1)).sum(): # 0.1 tolerance
        print('Warning! too big dt')

    raw_data_frame["vel_x"] = (raw_df_groupby["pos_x"].diff() / dt).astype(float)
    raw_data_frame["vel_y"] = (raw_df_groupby["pos_y"].diff() / dt).astype(float)
    nan_inds = np.array(np.nonzero(dt.isnull().to_numpy())).reshape(-1)
    raw_data_frame["vel_x"].iloc[nan_inds] = raw_data_frame["vel_x"].iloc[nan_inds + 1].to_numpy()
    raw_data_frame["vel_y"].iloc[nan_inds] = raw_data_frame["vel_y"].iloc[nan_inds + 1].to_numpy()
    
    raw_df_groupby = raw_data_frame.groupby("agent_id")
    
    trajs = [g for _, g in raw_df_groupby]
    return trajs

# In[6]:


def doOverlap(l1, r1, l2, r2):
     
    # To check if either rectangle is actually a line
      # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}
       
    if (l1.x == r1.x or l1.y == r1.y or l2.x == r2.x or l2.y == r2.y):
        # the line cannot have positive overlap
        return False
       
     
    # If one rectangle is on left side of other
    if(l1.x >= r2.x or l2.x >= r1.x):
        return False
 
    # If one rectangle is above other
    if(r1.y >= l2.y or r2.y >= l1.y):
        return False
 
    return True


# In[7]:


class GCSDatasetLoaderStatic(): 
    '''
    Create a static graph temporal signal of people at GCS
    - nodes: each node is a floor plan-based zone / room. 
    - node feature = avg speed, time. 
    - edges: unweighted. 1 = two connected zones (include diagonal). 
    '''
    def __init__(self, 
                 trajs,
                 ZONE_LIST):
        super(GCSDatasetLoaderStatic, self).__init__()
        self.trajs = trajs
        self.ZONE_LIST = ZONE_LIST
        self._read_data()
        
    def _read_data(self): 
        delta_t = trajs[0]['timestamp'].iloc[1]-trajs[0]['timestamp'].iloc[0]
        assert delta_t == 0.8
        all_trajs = pd.concat(trajs)
        all_trajs['speed'] = np.sqrt(all_trajs['vel_x']**2 + all_trajs['vel_y']**2)
        self.all_trajs = all_trajs 
        
        num_nodes = len(self.ZONE_LIST)
        A = np.ones((num_nodes,num_nodes))
        # todo: do connections automatically
        A -= np.eye(num_nodes)
        A[0:2,4:] = np.zeros_like(A[0:1,4:])
        A[2:4,6:] = np.zeros_like(A[2:4,6:])
        A[4:6,8]  = np.zeros_like(A[4:6,8])
        A = np.triu(A)
        A = A + A.T - np.diag(np.diag(A))
        
        # group by zones, each having a time vs speed df.
        zone_dfs = []
        common_time = [0, 1e6]
        for i in range(num_nodes): 
            x1, y1, x2, y2 = ZONE_LIST[i]
            tr = self.all_trajs[
                 (self.all_trajs['pos_x'] > x1) & 
                 (self.all_trajs['pos_x'] < x2) & 
                 (self.all_trajs['pos_y'] > y1) & 
                 (self.all_trajs['pos_y'] < y2)
                ]
            tr = tr.sort_values(by=['timestamp'])
#             # resample mean over 1 sec.
#             tr['time'] = pd.to_datetime(tr["timestamp"], unit='s')
#             tr.set_index('time').resample('1S')['vel_x'].mean().reset_index()
#             delta_t = 1
            
            temp_df = pd.DataFrame(columns=["timestamp", "speed", "num_people"])
            groupby = tr.groupby("timestamp", as_index=False)
            for _,g in groupby: 
                mean_speed = g['speed'].mean()
                num_people = g['agent_id'].nunique()
                timestamp = g['timestamp'].min()
                temp_df = temp_df.append(pd.DataFrame({"timestamp": [timestamp],
                                                       "speed": [mean_speed],
                                                       "num_people": [num_people]
                                                      }))
            tr = temp_df
            interp_t = np.arange(tr["timestamp"].iloc[0], tr["timestamp"].iloc[-1], delta_t)
            interp_func = interpolate.interp1d(tr["timestamp"], tr["speed"], kind='linear')
            interp_X_ = interp_func(interp_t)
            interp_func = interpolate.interp1d(tr["timestamp"], tr["num_people"], kind='linear')
            interp_Y_ = interp_func(interp_t)
            zone_dfs.append(pd.DataFrame({"timestamp": interp_t,
                                          "speed": interp_X_,
                                          "num_people": interp_Y_}))
            common_time[0] = max(common_time[0], interp_t.min())
            common_time[1] = min(common_time[1], interp_t.max())
            
        # crop out beginning and end with incomplete data
        for i, df in enumerate(zone_dfs):
            df = df.round({'timestamp':1})
            temp_df = df[(common_time[0] <= df['timestamp']) & 
                         (df['timestamp'] <= common_time[1])].copy()
            temp_df['time'] = (temp_df['timestamp'] - common_time[0])/(common_time[1]- common_time[0])
            zone_dfs[i] = temp_df[['num_people','time']].to_numpy().transpose([1,0])
        X = np.stack(zone_dfs)
        X = X.astype(np.float32)
        
        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values


    def _generate_task(self, num_timesteps_in, num_timesteps_out):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target
        
    def get_dataset(self, 
                    num_timesteps_in = 10, 
                    num_timesteps_out= 10): 
        '''10 step * 0.8 sec/step = 8 sec. '''
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset

# # Lightning Batch

# In[35]:


from torch_geometric_temporal.nn.recurrent import *
from torch_geometric_temporal.nn.attention import *

class A3TGCN_2(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(A3TGCN_2, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(in_channels=node_features,  out_channels=32, periods=periods,batch_size=batch_size) # node_features=2, periods=12
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index) # x [b, 207, 2, 12]  returns h [b, 207, 12]
        h = F.relu(h) 
#         h = F.dropout(h, training = self.training, p=0.5)
        h = self.linear(h)
        return h


# In[36]:


class BatchLitDataModule(pl.LightningDataModule):
    '''input shape (B, N, F, T)
       target shape (B,N,T)
       '''
    def __init__(self, loader, batch_size = 32, shuffle = True):
        super().__init__()
        self.save_hyperparameters("batch_size", "shuffle")
        self.loader = loader
        self.batch_size = batch_size
        self.shuffle=shuffle

    def setup(self, stage = None):
        dataset = self.loader.get_dataset(num_timesteps_in=200, num_timesteps_out=200)
        
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        
        train_input = np.array(train_dataset.features) # (27399, 207, 2, 12)
        train_edge_index = np.array([train_dataset.edge_index]*len(train_dataset.features))
        train_edge_weight = np.array([train_dataset.edge_weight]*len(train_dataset.features))
        train_target = np.array(train_dataset.targets) # (27399, 207, 12)
        train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor) # (B, N, F, T)
        train_edge_index_tensor = torch.from_numpy(train_edge_index)
        train_edge_weight_tensor = torch.from_numpy(train_edge_weight)
        train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor)  # (B, N, T)
        self.train_dataset = torch.utils.data.TensorDataset(train_x_tensor, 
                                                            train_edge_index_tensor, 
                                                            train_edge_weight_tensor,
                                                            train_target_tensor)
        
        test_input = np.array(test_dataset.features) # (, 207, 2, 12)
        test_edge_index = np.array([test_dataset.edge_index]*len(test_dataset.features))
        test_edge_weight = np.array([test_dataset.edge_weight]*len(test_dataset.features))
        test_target = np.array(test_dataset.targets) # (, 207, 12)
        test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor)  # (B, N, F, T)
        test_edge_index_tensor = torch.from_numpy(test_edge_index)
        test_edge_weight_tensor = torch.from_numpy(test_edge_weight)
        test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)
        self.test_dataset = torch.utils.data.TensorDataset(test_x_tensor, 
                                                           test_edge_index_tensor, 
                                                           test_edge_weight_tensor,
                                                           test_target_tensor)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           drop_last=True,
                                           num_workers=4)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=True,
                                           num_workers=4)

class BatchLitWrapper(pl.LightningModule): 
    def __init__(self, model, lr): 
        super().__init__()
        self.save_hyperparameters("lr")
        self.model = model
        self.lr = lr
        
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.2,
                                                                    verbose=True)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
    
    def _shared_step(self, batch, batch_idx): 
        x, edge_index, edge_attr, labels = batch
        edge_index = edge_index[0,:,:] # because its static
        edge_attr = edge_attr[0,:]
        h = self.model(x, edge_index)
        return h, labels
    
    def training_step(self, batch, batch_idx): 
        h, y = self._shared_step(batch, batch_idx)
        loss = self._get_loss(h, y)
        metrics = {'train_loss': loss,
                   'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(metrics)
        return loss
        
    def validation_step(self, batch, batch_idx):
        h, y = self._shared_step(batch, batch_idx)
        loss = self._get_loss(h, y)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx) :
        h, y = self._shared_step(batch, batch_idx)
        loss = self._get_loss(h, y)
        metrics = {'test_loss': loss}
        self.log_dict(metrics)
        return metrics
        
    def _get_loss(self, h, y): 
        return F.mse_loss(h, y)

if __name__ == "__main__":
    
    path = "dataset/gcs/Annotation"
    trajs = parse_gcs(path) # traj_datasets = load_gcs(path)


    # 7 zones. each zone (x1, y1, x2, y2)
    ZONE_LIST =[
        (0, 0, 28, 14),
        (28, 0, 55, 14),
        (0, 14, 28, 24),
        (28, 14, 55, 24),
        (0, 24, 28, 35),
        (28, 24, 55, 35),
        (0, 35, 28, 45),
        (28, 35, 55, 45),
        (0, 45, 55, 55)
    ]
    loader = GCSDatasetLoaderStatic(
        trajs=trajs,
        ZONE_LIST = ZONE_LIST)

    for lr in [1e-2]:
        for batch_size in [32]: # 1,2,4,8,16,32,64,128,256,512
            datamodule = BatchLitDataModule(loader, 
                                            batch_size = batch_size, 
                                            shuffle=True)
            datamodule.setup()

            for m in [A3TGCN_2(node_features=2, periods=200, batch_size=batch_size),
                     ]: 
                
                model = BatchLitWrapper(m, lr)
                
                logger = pl.loggers.TensorBoardLogger(
                    save_dir = "./lightning_logs",
                    name = m.__class__.__name__)
                
                early_stop_callback = EarlyStopping(monitor='val_loss',
                                    min_delta=0.00,
                                    patience=100,
                                    verbose=True,
                                    mode='min') # why the hell is this max 
                
                checkpoint_callback = ModelCheckpoint(save_last=True,
                                                      verbose=True, 
                                                      monitor='val_loss',
                                                      mode='min')


                trainer = pl.Trainer(callbacks=[early_stop_callback,
                                                checkpoint_callback],
                                     logger = logger,
                                     gpus=1,
                                     accelerator='gpu',
    #                                  fast_dev_run=10
                                    )
                trainer.fit(model, datamodule=datamodule)

# In[34]:


# # testing
# batch_size = 1
# datamodule = BatchLitDataModule(loader, 
#                                 batch_size = batch_size, 
#                                 shuffle=True)
# datamodule.setup()

# for m in [A3TGCN_2(node_features=2, periods=20, batch_size=batch_size),
#          ]: 
#     PATH = "/version_10/checkpoints/epoch=15-step=66207.ckpt"
#     model = BatchLitWrapper.load_from_checkpoint("lightning_logs/"+m.__class__.__name__+PATH,
#                                                  model=m)
#     logger = pl.loggers.TensorBoardLogger(save_dir = "./lightning_logs/flow_forecasting",name=m.__class__.__name__)
#     trainer = pl.Trainer(callbacks=[early_stop_callback,
#                                     checkpoint_callback],
#                          logger = logger,
#                          gpus=1,
#                          accelerator='gpu',
# #                                  fast_dev_run=10
#                         )
#     trainer.test(model, datamodule=datamodule)


# In[38]:


# trainer = pl.Trainer(callbacks=[early_stop_callback],
#                          auto_lr_find=True,
#                          logger = logger,
#                          gpus=1,
#                          accelerator='gpu')
# trainer.tune(model, datamodule=datamodule)


# In[ ]:




