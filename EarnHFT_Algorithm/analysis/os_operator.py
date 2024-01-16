import pandas as pd
import numpy as np
import os
import re
import shutil

root_path="result_risk"
for dataset in os.listdir(root_path):
    dataset_path=os.path.join(root_path,dataset)
    for para in ["beta_-10.0_risk_bond_0.1","beta_-90.0_risk_bond_0.1","beta_30.0_risk_bond_0.1","beta_100.0_risk_bond_0.1"]:
        para_path=os.path.join(dataset_path,para,"seed_12345")
        epoch_list=os.listdir(para_path)
        for epoch in epoch_list:
            epoch_path=os.path.join(para_path,epoch)
            if os.path.exists(os.path.join(epoch_path,"valid_multi")):
                shutil.rmtree(os.path.join(epoch_path,"valid_multi"))
            