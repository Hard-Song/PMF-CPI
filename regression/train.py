from model import *
from utils import *
import numpy as np
from sklearn.model_selection import KFold
import torch
import os
import warnings
warnings.filterwarnings("ignore")


dataset = ['toy_reg','BindingDB_reg','JAK','AR','CYP_reg'][int(sys.argv[1])]
TRAIN_BATCH_SIZE =32
TEST_BATCH_SIZE = 256
LR = 0.0001
NUM_EPOCHS = 5
seed = 0
dropout_global = 0.2
train_val_ratio = 0.9
early_stop = 20
stop_epoch = 0
best_epoch = -1
best_mse = 100
best_test_mse = 100
best_epoch = -1
last_epoch = 1


# choose embedding 
emb_type = int(sys.argv[2])
cuda_name = ['cuda:0','cuda:1','cuda:2','cuda:3'][0]
protein_embedding = ['onehot', 'bio2vec', 'tape', 'esm2'][emb_type]
emb_size = [20, 100, 768, 1280][emb_type]
parmeter =  'ratio_' +str(train_val_ratio)+'bach'+str(TRAIN_BATCH_SIZE) + 'LR_1e-4'+'random_0_' + protein_embedding
# Path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
dataset_path = "../datasets/" + dataset + '/train/'
model_file_dir = dataset + '/'
embedding_path = "../datasets/%s/%s/"%(dataset,protein_embedding)
# set protein length capacity, which default=1500(according to BindingDB regression dataset)
max_length = max(1500,int(open("../datasets/" + dataset + '/max_length.txt','r').read()))
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
model = CPI_regression(device,emb_size=emb_size,max_length=max_length,dropout=dropout_global)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=1)


# load all entries to generate train_val set
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
print('epoch\ttrain_loss\tval_loss')
for epoch in range(NUM_EPOCHS):
    train_loss = train(model, device, train_loader, optimizer, epoch + 1, writer)
    val_loss, T, P = evaluate(model, device, valid_loader)
    writer.add_scalar('Valid/Loss', val_loss, epoch)
    print(epoch+1,'\t',train_loss,'\t',val_loss)
    stop_epoch+=1
    if val_loss < best_mse:
        best_mse = val_loss
        stop_epoch = 0
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_name)
    if stop_epoch == early_stop:
        print('(EARLY STOP) No improvement since epoch ', best_epoch, '; best_test_mse', best_mse, dataset)
        break
    scheduler.step(val_loss)
print('Best epoch %s; best_test_mse%s; dataset:%s; train ratio:%s'%(best_epoch, best_mse, dataset,train_val_ratio))
    
