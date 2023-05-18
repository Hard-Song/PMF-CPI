from model import *
from utils import *
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings("ignore")

TEST_BATCH_SIZE =256

model_name = 'ratio_0.9bach128LR_1e-4random_0_onehot.pt'


dataset = ['tmp','BindingDB_reg','JAK','AR','CYP_reg'][int(sys.argv[1])]
cuda_name = ['cuda:0','cuda:1','cuda:2','cuda:3'][0]
emb_type = int(sys.argv[2])

protein_embedding = ['onehot', 'bio2vec', 'tape', 'esm2'][emb_type]
emb_size = [20, 100, 768, 1280][emb_type]

# instantiate a model
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
print(device)
embedding_path = "../datasets/%s/%s/"%(dataset,protein_embedding)
max_length = max(1500,int(open("../datasets/" + dataset + '/max_length.txt','r').read()))
model = CPI_regression(device,emb_size=emb_size,max_length=max_length,dropout=0)
model.to(device)


# Path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
dataset_path = "../datasets/" + dataset + '/ind_test/'
test_fold = eval(open(dataset_path+'valid_entries.txt','r').read())
embedding_path = "../datasets/%s/%s/"%(dataset,protein_embedding)
if not os.path.exists(embedding_path):
    print("No protein embedding files, please generate relevant embedding first!")
    exit(0)

model.load_state_dict(torch.load(dataset + '/'+ model_name, map_location=device), strict=False)
test_data = create_dataset_for_test(dataset_path,embedding_path,test_fold)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,collate_fn=collate)
T, P = predicting(model, device, test_loader)
mse = get_mse(T, P)
pearson = get_pearson(T, P)
print(dataset,'\tmse:', mse,'\tpearson', pearson)
np.save('%s/True.npy'%(dataset),T)
model_name = model_name.replace('.pt','')
np.save('%s/Predict_%s.npy'%(dataset,model_name),P)


