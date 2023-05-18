
import sys, os
import time
import torch
import json, pickle
import numpy as np
import pandas as pd
from math import sqrt
import networkx as nx
from collections import OrderedDict
from rdkit import Chem
from scipy import stats
from rdkit.Chem import MolFromSmiles
from torch_geometric import data as DATA
from torch.utils.data.dataloader import default_collate
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
import math


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None,
                 xd=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, target_key=None, target_graph=None):

        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_graph)
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_graph):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'These lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_list_pro_len = []
        data_len = len(xd)
        for i in range(data_len):
            smiles = xd[i]
            tar_key = target_key[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            target_features, target_size= target_graph[tar_key]
            GCNData_mol = DATA.Data(x=torch.Tensor(features),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels]))
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(target_features)
            data_list_pro_len.append(target_size)

        self.data_mol = data_list_mol
        self.data_pro = data_list_pro
        self.data_pro_len = data_list_pro_len

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx], self.data_pro_len[idx] 


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    train_loss = []
    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        data_pro_len = data[2].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro, data_pro_len)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
    train_loss.append(loss.item())
    train_loss = np.average(train_loss)
    writer.add_scalar('Train/Loss', train_loss, epoch)
    return train_loss

def evaluate(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    loss_fn = torch.nn.MSELoss()
    eval_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            output = model(data_mol, data_pro, data_pro_len)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
            # sys.stdout.write(str(format(100. *batch_idx / len(loader),'.2f'))+"%\r")
        eval_loss.append(loss.item())
    eval_loss = np.average(eval_loss)
    return eval_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()



# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            data_pro_len = data[2].to(device)
            output = model(data_mol, data_pro, data_pro_len)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
            # sys.stdout.write(str(format(100. *batch_idx / len(loader),'.2f'))+"%\r")
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


#prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = default_collate([data[1] for data in data_list])
    batchC = default_collate([data[2] for data in data_list])
    return batchA, batchB, batchC 




# nomarlize

# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index


def target_matrics(key,embedding_path):
    eembedding_file = os.path.join(embedding_path, key)
    input_feature = torch.load(eembedding_file)
    return input_feature['feature'], input_feature['size']


def create_dataset_for_test(dataset_path,embedding_path,test_fold):
    # load dataset
    ligands = json.load(open(dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    smile_file_name = dataset_path + '/smile_graph'
    
    # load compounds graph 
    compound_iso_smiles = []
    smile_graph = {}
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        compound_iso_smiles.append(lg)
    if os.path.exists(smile_file_name):
        print('load smile graph ...')
        smile_graph = pickle.load(open(smile_file_name, 'rb'))
    else:
        # create smile graph
        print('create smile_graph ...')
        smile_graph = {}
        for smile in compound_iso_smiles:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        with open(smile_file_name,'wb+') as f:
            pickle.dump(smile_graph,f)
    # load seqs
    target_key = []
    target_graph = {}
    for key in proteins.keys():
        target_key.append(key)
        target_graph[key] = target_matrics(key,embedding_path)
    # load affinity matrix...
    print('load affinity matrix...')
    # pick out test entries
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[test_fold], cols[test_fold]
    test_fold_entries = {'compound_iso_smiles':[],'target_key':[],'affinity':[]}
    for pair_ind in range(len(rows)):
        test_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[rows[pair_ind]]]
        test_fold_entries['target_key']  += [target_key[cols[pair_ind]]]
        test_fold_entries['affinity']  += [affinity[rows[pair_ind], cols[pair_ind]]]
    test_fold = pd.DataFrame(test_fold_entries)
    test_drugs, test_prot_keys, test_Y = np.asarray(list(test_fold['compound_iso_smiles'])), np.asarray(list(test_fold['target_key'])), np.asarray(list(test_fold['affinity']))
    test_dataset = DTADataset(root='../data', dataset= dataset_path + '_test', xd=test_drugs, y=test_Y,
                              target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph)
    return test_dataset


def create_dataset_for_train(dataset_path,embedding_path,train_valid):
    # load dataset
    ligands = json.load(open(dataset_path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    smile_file_name = dataset_path + '/smile_graph'
    # load compounds graph 
    compound_iso_smiles = []
    smile_graph = {}
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        compound_iso_smiles.append(lg)
    if os.path.exists(smile_file_name):
        print('load smile graph ...')
        smile_graph = pickle.load(open(smile_file_name, 'rb'))
    else:
        # create smile graph
        print('create smile_graph ...')
        smile_graph = {}
        for smile in compound_iso_smiles:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        with open(smile_file_name,'wb+') as f:
            pickle.dump(smile_graph,f)
    # load seqs
    target_key = []
    target_graph = {}
    for key in proteins.keys():
        target_key.append(key)
        target_graph[key] = target_matrics(key,embedding_path)
    # load affinity matrix...
    print('load affinity matrix...')
    train_fold = train_valid[0]
    valid_fold = train_valid[1]
    print('train entries:', len(train_fold))
    print('valid entries:', len(valid_fold))
    stime=time.time()
    print('load train data...')
    rows, cols = np.where(np.isnan(affinity) == False)
    trows, tcols = rows[train_fold], cols[train_fold]  
    train_fold_entries = {'compound_iso_smiles':[],'target_key':[],'affinity':[]}
    for pair_ind in range(len(trows)):   
        train_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[trows[pair_ind]]]  
        train_fold_entries['target_key']  += [target_key[tcols[pair_ind]]]
        train_fold_entries['affinity']  += [affinity[trows[pair_ind], tcols[pair_ind]]]  
    print('load valid data...')
    trows, tcols = rows[valid_fold], cols[valid_fold]
    valid_fold_entries = {'compound_iso_smiles':[],'target_key':[],'affinity':[]}
    for pair_ind in range(len(trows)):
        valid_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[trows[pair_ind]]]
        valid_fold_entries['target_key'] += [target_key[tcols[pair_ind]]]
        valid_fold_entries['affinity'] += [affinity[trows[pair_ind], tcols[pair_ind]]]
    print('done time consuming:',time.time()-stime)
    df_train_fold = pd.DataFrame(train_fold_entries)
    train_drugs, train_prot_keys, train_Y = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['affinity'])
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)
    train_dataset = DTADataset(root=dataset_path, dataset=dataset_path + '_train', xd=train_drugs, target_key=train_prot_keys,
                               y=train_Y, smile_graph=smile_graph, target_graph=target_graph)


    df_valid_fold = pd.DataFrame(valid_fold_entries)
    valid_drugs, valid_prots_keys, valid_Y = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_key']), list(df_valid_fold['affinity'])
    valid_drugs, valid_prots_keys, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots_keys), np.asarray(
        valid_Y)
    valid_dataset = DTADataset(root=dataset_path, dataset=dataset_path + '_valid', xd=valid_drugs,
                               target_key=valid_prots_keys, y=valid_Y, smile_graph=smile_graph,
                               target_graph=target_graph)
    return train_dataset, valid_dataset



#############  Evaluate matrix   #############
#############  Evaluate matrix   #############
def get_cindex(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

