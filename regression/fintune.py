from model import *
from utils import *
import numpy as np
from sklearn.model_selection import KFold
import torch
import os,sys
import warnings
warnings.filterwarnings("ignore")


# trained_model protein length capacity should >= new protein length capacity
# trained_model should have same embedding method with new dataset
dataset = ["toy_reg","AR","JAK","CYP_reg"][int(sys.argv[1])]
ratios = [0.05,0.1,0.15,0.2,0.25,0.3]
trained = 'trained.pt'
TRAIN_BATCH_SIZE =32
TEST_BATCH_SIZE = 256
LR = 0.0001
NUM_EPOCHS = 2
seed = 0
dropout_global = 0.2
early_stop = 20

# choose embedding method
emb_type = int(sys.argv[2])
cuda_name = ['cuda:0','cuda:1','cuda:2','cuda:3'][0]
protein_embedding = ['onehot', 'bio2vec', 'tape', 'esm2'][emb_type]
emb_size = [20, 100, 768, 1280][emb_type]
parmeter = '_fintune_'+ protein_embedding
# Path=os.path.abspath(os.path.join(os.getcwd(),"../.."))
dataset_path = "../datasets/" + dataset + '/train/'
model_file_dir = dataset + parmeter + '/'
embedding_path = "../datasets/%s/%s/"%(dataset,protein_embedding)
# instantiate a model
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')
# set protein length capacity, which default=1500(according to BindingDB regression dataset)
max_length = max(1500,int(open("../datasets/" + dataset +'/max_length.txt','r').read()))
model = CPI_regression(device,emb_size=emb_size,max_length=max_length,dropout=dropout_global)
model.to(device)
model.load_state_dict(torch.load(trained, map_location=device), strict=False)
print(f'fintune on model:{trained};device:{device};')


# load all entries to generate train_val_test fold
kFold = KFold(n_splits=5, shuffle=True)
raw_fold = eval(open(dataset_path + 'valid_entries.txt','r').read())
np.random.seed(seed)
random_entries = np.random.permutation(raw_fold)
for ratio in ratios:
    train_ptr = max(10,int(len(random_entries)*ratio))  # train entries must >=10
    this_train_dataset = random_entries[:train_ptr]
    for fold, (train_index, val_index) in enumerate(kFold.split(this_train_dataset)):
        train_entry = this_train_dataset[train_index]
        val_entry = this_train_dataset[val_index]
        this_train_val = [train_entry.tolist(),val_entry.tolist()]
        print(f"dataset: {dataset} ratio:{ratio} fold: {fold}")
        if not os.path.exists(model_file_dir):
            os.makedirs(model_file_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, verbose=1)
        train_data, valid_data = create_dataset_for_train(dataset_path,embedding_path,this_train_val)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                                    collate_fn=collate)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                    collate_fn=collate)
        
        # train
        model_name = model_file_dir + str(ratio)+'_' + str(fold) + '.pt'
        log_dir =  dataset+'/logs/' + parmeter  + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name =  "ratio_" +str(ratio) + "_fold_" + str(fold) 
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=log_file_name)
        stop_epoch = 0
        best_epoch = -1
        best_mse = 100
        best_test_mse = 100
        best_epoch = -1
        last_epoch = 1
        print('dataset\tratio\tfold\tepoch\ttrain_loss\tval_loss')
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, device, train_loader, optimizer, epoch + 1, writer)
            val_loss, T, P = evaluate(model, device, valid_loader)
            writer.add_scalar('Valid/Loss', val_loss, epoch)
            print(dataset,'\t',ratio,'\t',fold,'\t',epoch,'\t',train_loss,'\t',val_loss)
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
        print('Best epoch ', best_epoch, '; best_test_mse', best_mse, dataset,ratio,fold)
            
