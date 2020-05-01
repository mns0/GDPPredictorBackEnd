# get the data
import numpy as np
import torch
import app.models_and_dataloader 
from app.models_and_dataloader import GDPData
from app.models_and_dataloader import Generator_RNN
from app.models_and_dataloader import LockedDropout
import pickle

def make_prediction():
    dir_data = './app/'
    train_data = 'Preprocessed_data_gdp_pc.dat'
    start, end, stride = 0, 2000, 100
    epoches, batch_size, seq_len = 200, 20, 15
    
    #kwagsG = {"target_size": 1, "predictor_dim": 10, "hidden_dim": 200,
    #          "num_layers": 1, "dropout_prob": 0.4, 'train': False}

    kwagsG = {"target_size": 1, "predictor_dim": 10, "hidden_dim": 300,
              "num_layers": 5, "dropout_prob": 0.5, 'train': True}

    device = torch.device('cpu')
    gdp_dataset = GDPData(dir_data + train_data, normalize=True)
    x, y  = gdp_dataset.x.squeeze(-1), gdp_dataset.y.unsqueeze(2)
    _min, _max = gdp_dataset.normalize_table[-1]
   
    #get generator data
    generated_time_series = []
    i = 500 
    gen_model = f'alt_netG_e{i}.pth'
    check_gen = Generator_RNN(**kwagsG)
    print("Loading: {}".format(dir_data+gen_model) )
    check_gen.load_state_dict(torch.load(dir_data+gen_model,
        map_location=device))

    check_gen.eval()
    gen_ts = []
    end_idx = (len(x)//batch_size)*batch_size
    #updates prediction for whole array
    for cond in np.array_split(x[:end_idx],len(x)//batch_size):
      if cond.shape[0] != batch_size:
        continue
      noise_x = torch.randn(batch_size, seq_len, 1)
      gen_data = check_gen(noise_x,cond).detach().numpy().reshape(-1)
      gen_ts.extend(gen_data.reshape(-1))
    
    #unnormalize data
    gen_ts = np.asarray(gen_ts)
    gen_ts = 0.5* (gen_ts*_max - gen_ts*_min + _max + _min)
    
    return gdp_dataset.dates[:len(gen_ts)], gen_ts
