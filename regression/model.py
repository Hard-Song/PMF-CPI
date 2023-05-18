import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_max_pool as gmp, global_mean_pool as gep



class CPI_regression(torch.nn.Module):
    def __init__(self, device,emb_size, max_length, n_output=1, hidden_size=128, num_features_mol=78,dropout=None):
        super(CPI_regression, self).__init__()
        print('CPI_regression model Loaded..')
        self.device = device
        self.skip = 1
        self.n_output = n_output
        self.max_length = max_length
        # compounds network
        self.mol_conv1 = SAGEConv(num_features_mol, num_features_mol*2,'mean')
        self.mol_conv2_f = SAGEConv(num_features_mol*2,num_features_mol*2,'mean')
        self.mol_conv3_f = SAGEConv(num_features_mol*2,num_features_mol*4,'mean')
        self.mol_fc_g1 = nn.Linear(num_features_mol*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # proteins network
        self.prot_rnn = nn.LSTM(emb_size, hidden_size, 1)
        self.relu = nn.LeakyReLU()
        # combined layers
        self.prot_comp_mix = nn.Sequential(nn.Linear(129 * 129, 1024), nn.LeakyReLU(), nn.Dropout(dropout))
        self.fc = nn.Sequential(nn.Linear(1282, 512),nn.LeakyReLU(),nn.Dropout(dropout),
                                nn.Linear(512, self.n_output))


    def forward(self, data_mol, data_pro, data_pro_len):
        # protein network
        pro_seq_lengths, pro_idx_sort = torch.sort(data_pro_len,descending=True)[::-1][1], torch.argsort(
            -data_pro_len)
        pro_idx_unsort = torch.argsort(pro_idx_sort)
        data_pro = data_pro.index_select(0, pro_idx_sort)
        xt = nn.utils.rnn.pack_padded_sequence(data_pro, pro_seq_lengths.cpu(), batch_first=True)
        xt,_ = self.prot_rnn(xt)
        xt = nn.utils.rnn.pad_packed_sequence(xt, batch_first=True, total_length=max(1500,self.max_length))[0]
        xt = xt.index_select(0, pro_idx_unsort)
        xt = xt.mean(1)
        
        # compound network
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv2_f(x, mol_edge_index)
        x = self.relu(x)
        x = self.mol_conv3_f(x, mol_edge_index)
        x = self.relu(x)
        x = gmp(x, mol_batch)  # global max pooling
        x = self.mol_fc_g1(x)
        x = self.dropout(x)

        # kronecker product
        prot_out = torch.cat((xt, torch.FloatTensor(xt.shape[0], 1).fill_(1).to(self.device)), 1) 
        comp_out = torch.cat((x, torch.FloatTensor(x.shape[0], 1).fill_(1).to(self.device)), 1)  
        output = torch.bmm(prot_out.unsqueeze(2), comp_out.unsqueeze(1)).flatten(start_dim=1) 
        output = self.dropout(output)
        output = self.prot_comp_mix(output)
        if self.skip:
            output = torch.cat((output, prot_out, comp_out), 1)
        output = self.fc(output)
        return output

