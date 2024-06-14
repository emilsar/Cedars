import torch, glob
import pandas as pd, numpy as np
if __name__=="__main__":
    urothelial_cells=dict(X=torch.stack([torch.tensor(np.load(f"imagedata/X/{i}.npy")) for i in range(len(glob.glob("imagedata/X/*.npy")))],axis=0),
                            y=np.stack([np.load(f"imagedata/y/{i}.npy") for i in range(len(glob.glob("imagedata/y/*.npy")))],axis=0))
    pd.to_pickle(urothelial_cells,"urothelial_cell_toy_data.pkl")