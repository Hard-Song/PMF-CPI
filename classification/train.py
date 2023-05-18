from model import *
from utils import *
import numpy as np
from sklearn.model_selection import KFold
import torch
import os,sys
import warnings
warnings.filterwarnings("ignore")


dataset = ['toy_cls','BindingDB_cls','DrugAI','CYP_cls'][int(sys.argv[1])]
TRAIN_BATCH_SIZE =32
TEST_BATCH_SIZE = 256
LR = 0.0001
NUM_EPOCHS = 2
seed = 0
dropout_global = 0.2
train_val_ratio = 0.9
early_stop = 20
stop_epoch = 0
best_epoch = -1
best_auc = 0.5
last_epoch = 1


# choose embedding 
emb_type = int(sys.argv[1])
cuda_name = ['cuda:0','cuda:1','cuda:2','cuda:3'][0]
protein_embedding = ['onehot', 'bio2vec', 'tape', 'esm2'][emb_type]
emb_size = [20, 100, 768, 1280][emb_type]
parmeter =  'ratio_' +str(train_val_ratio)+'bach'+str(TRAIN_BATCH_SIZE) + 'LR_1e-4'+'random_0_' + protein_embedding
# Path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
dataset_path = "../datasets/" + dataset + '/train/'
model_file_dir = dataset + '/'
embedding_path = "../datasets/%s/%s/"%(dataset,protein_embedding)
# set protein length capacity, which default=4680(according to BindingDB classification dataset)
max_length = max(4680,int(open("../datasets/" + dataset +'/max_length.txt','r').read()))
model_name = model_file_dir + parmeter + '.pt'
log_dir = model_file_dir+ 'logs/' 
writer = SummaryWriter(log_dir=log_dir, filename_suffix=parmeter)
if not os.path.exists(model_file_dir):
    os.makedirs(model_file_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# instantiate a model
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
model = CPI_classification(device,emb_size=emb_size,max_length=max_length,dropout=dropout_global)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=1)


# load all valid entries to generate train_val set
raw_fold = eval(open(dataset_path + 'valid_entries.txt','r').read())
np.random.seed(seed)
random_entries = np.random.permutation(raw_fold)
ptr = int(train_val_ratio*len(random_entries))
train_val = [random_entries[:ptr],random_entries[ptr:]]
train_data, valid_data = create_dataset_for_train(dataset_path, embedding_path, train_val)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                            collate_fn=collate)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                            collate_fn=collate)

# train
print('epoch\ttime\ttrain_loss\tval_loss\tAUROC\tAUPRC')
for epoch in range(NUM_EPOCHS):
    time_start = time.time()
    train_loss = train(model, device, train_loader, optimizer, epoch + 1, writer)
    T, S, val_loss = evaluate(model, device, valid_loader)
    writer.add_scalar('Valid/Loss', val_loss, epoch)
    S = S[:, 1]
    P = (S > 0.5).astype(int)
    AUROC = roc_auc_score(T, S)
    AUCS = [str(epoch+1),str(format(time.time()-time_start, '.1f')),str(format(train_loss, '.4f')),str(format(val_loss, '.4f')),str(format(AUROC, '.4f'))]
    print('\t'.join(map(str, AUCS)))
    if AUROC >= best_auc:
        stop_epoch = 0
        best_auc = AUROC
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_name)
    else:
        stop_epoch += 1
    if stop_epoch == early_stop:
        print('(EARLY STOP) No improvement since epoch ', best_epoch, '; best_test_AUC', best_auc)
        break