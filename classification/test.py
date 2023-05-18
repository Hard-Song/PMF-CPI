from model import *
from utils import *
import numpy as np
import torch
import os,sys
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc,accuracy_score,f1_score


TEST_BATCH_SIZE =256
model_name = 'ratio_0.9bach32LR_1e-4random_0_onehot.pt'
dataset = ['toy_cls','BindingDB_cls','DrugAI','CYP_cls'][int(sys.argv[1])]
cuda_name = ['cuda:0','cuda:1','cuda:2','cuda:3'][0]
emb_type = int(sys.argv[2])
protein_embedding = ['onehot', 'bio2vec', 'tape', 'esm2'][emb_type]
emb_size = [20, 100, 768, 1280][emb_type]
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
print(device)

embedding_path = "../datasets/%s/%s/"%(dataset,protein_embedding)
# set protein length capacity, which default=1500(according to BindingDB classification dataset)
max_length = max(4680,int(open("../datasets/" + dataset +'/max_length.txt','r').read()))
model = CPI_classification(device,emb_size=emb_size,max_length=max_length,dropout=0)
model.to(device)


# Path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
dataset_path = "../datasets/" + dataset + '/ind_test/'
test_fold = eval(open(dataset_path+'valid_entries.txt','r').read())
if not os.path.exists(embedding_path):
    print("No protein embedding files, please generate relevant embedding first!")
    exit(0)
model.load_state_dict(torch.load(dataset + '/'+ model_name, map_location=device), strict=False)
test_data = create_dataset_for_test(dataset_path,embedding_path,test_fold)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,collate_fn=collate)
T, S = predicting(model, device, test_loader)
S = S[:, 1]
P = (S > 0.5).astype(int)
AUROC = roc_auc_score(T, S)
tpr, fpr, _ = precision_recall_curve(T, S)
AUPR = auc(fpr, tpr)
ACC = accuracy_score(T,P)
REC = recall_score(T,P)
f1 = f1_score(T, P)
AUCS_key = ['AUC','PRC','ACC','Recall','f1']
AUCS = [AUROC,AUPR,ACC,REC,f1]
print('\t'.join(AUCS_key))
print('\t'.join([f"{num:.3f}" for num in AUCS]))
np.save('%s/True.npy'%(dataset),T)
model_name = model_name.replace('.pt','')
np.save('%s/Predict_%s.npy'%(dataset,model_name),P)


