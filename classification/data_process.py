import pickle
import json
import numpy as np
import os,sys
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
import biovec
from Bio.Seq import Seq
from biovec import models
from tape import ProteinBertModel, TAPETokenizer
import warnings
warnings.filterwarnings("ignore")



def generate_Y(dataset):
    path = dataset + '/'
    affinity = open(path+'affinity.tsv','r').readlines()
    dict_prot_thisSet = json.load(open(path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    dict_comp_thisSet = json.load(open(path + 'compounds.txt'), object_pairs_hook=OrderedDict)
    print("generate Y by dataset:",dataset)
    rows,cols = len(dict_comp_thisSet.keys()),len(dict_prot_thisSet.keys())
    print("effictive Drugs,proteins:",rows,cols)
    y = np.full((rows,cols),np.nan)
    aff_matrx = []
    for i in affinity:
          tmp_list = i.strip().split('\t')
          try:
            tmp_list[2] = int(tmp_list[2])
          except:
              print(tmp_list,'format should belike: protein_key \t compound_key \t affinity')
              exit(0)
          aff_matrx.append(tmp_list)
    prot_col = [i for i in dict_prot_thisSet.keys()]
    comp_row = [i for i in dict_comp_thisSet.keys()]
    count=0
    same_entry = []
    same_index = []
    print("create affinity matrix...")
    
    for ptr in range(len(aff_matrx)):
        col = prot_col.index(aff_matrx[ptr][0])
        row = comp_row.index(aff_matrx[ptr][1])
        if np.isnan(y[row][col]):
            count +=1
            y[row][col]= aff_matrx[ptr][2]
        else:
            same_index.append(ptr)
            same_entry.append([row,col])
            y[row][col]+= aff_matrx[ptr][2]

    # For regression, calculate avg of duplicated data
    while same_entry:
        n=2
        index = same_entry.pop()
        while index in same_entry:
            same_entry.remove(index)
            n+=1
        row = index[0]
        col = index[1]
        y[row][col] = int(y[row][col]/n)
    print('writing to local file...')
    yyy = open(path+'Y','wb')
    print(" dataset:",dataset,"finished; raw entries:",len(affinity),"entries:",count)
    pickle.dump(y,yyy)
    return count

def generate_fold(dataset_path):
    valid_entries = generate_Y(dataset_path)
    valid_index = [i for i in range(valid_entries)]
    with open(dataset_path+'/valid_entries.txt','w') as f:
            print(valid_index,file=f)
    print(dataset_path,'valid entries:',valid_entries)



def seq_to_kmers(seq, k=3):
    N = len(seq)
    return [seq[i:i+k] for i in range(N - k + 1)]



def onehot(sequence_dict,out_file_path,max_length):
    Alfabeto = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    count = 0
    for key in sequence_dict.keys():
        sequence = sequence_dict[key]
        count += 1
        feature = np.zeros(shape=[max_length, len(Alfabeto)],dtype='float32')
        sequence = sequence.upper()
        size = len(sequence)
        indices = [Alfabeto.index(c) for c in sequence if c in Alfabeto]
        for j, index in enumerate(indices):
            feature[j, index] = float(1.0)
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count}/{len(sequence_dict)})', end='')
        feature = torch.from_numpy(feature)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+key)


def Bio2vec(sequence_dict,out_file_path,max_length):
    fasta_txt = ''
    for key in sequence_dict.keys():
        fasta_txt = fasta_txt + '>' + key + '\n'+ sequence_dict[key] + '\n'
    with open(out_file_path+"/bio2vec.fasta",'w') as f:
        print(fasta_txt.strip(),file=f)
    pv = biovec.models.ProtVec(out_file_path+ "/bio2vec.fasta", corpus_fname=out_file_path+ "/bio2vec_corpus.txt", n=3)
    count = 0
    for seq in sequence_dict.keys():
        sequence = seq_to_kmers(sequence_dict[seq], k=3)
        size = len(sequence)
        vec = np.zeros((max_length, 100),dtype='float32')
        i = 0
        for word in sequence:
            vec[i] = pv.to_vecs(word)[0]
            i += 1
        feature = torch.from_numpy(vec)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+ seq)
        count+=1
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequence_dict)})', end='')


def tape_embedding(sequence_dict,out_file_path,max_length):
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
    model.eval()
    count = 0
    for key in sequence_dict.keys():
        tmp_list = [tokenizer.encode(sequence_dict[key].upper())]
        tmp_array = np.array(tmp_list)
        token_ids = torch.from_numpy(tmp_array)  # encoder
        sequence_output, _ = model(token_ids)   
        sequence_output = sequence_output.detach().numpy()
        # padding to same size [???,768]
        feature = sequence_output.squeeze()
        feature = np.delete(feature,-1,axis=0)
        feature = np.delete(feature,0,axis=0)
        size = feature.shape[0]
        pad_length = max_length - size
        if pad_length:
            padding = np.zeros((pad_length,768),dtype='float32')
            feature = np.r_[feature,padding]
        feature = torch.from_numpy(feature)
        embeddings = {"feature":feature, "size":size}
        torch.save(embeddings, out_file_path + '/'+key)
        count+=1
        percent = int((count / len(sequence_dict)) * 100)
        bar = '#' * int(count / len(sequence_dict) * 20)
        print(f'\r[{bar:<20}] {percent:>3}% ({count+1}/{len(sequence_dict)})', end='')
    


def generate_embeddings(dataset,embedding_type):
    sequence_dict = eval(open(dataset+'/proteins.txt','r').read())
    out_file_path = dataset + '/' + embedding_type
    max_length = 0
    for key in sequence_dict.keys():
        max_length = max(max_length,len(sequence_dict[key])+1)
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if embedding_type=='tape':
        tape_embedding(sequence_dict,out_file_path,max_length)
    elif embedding_type=='onehot':
        onehot(sequence_dict,out_file_path,max_length)
    elif embedding_type=='bio2vec':
        Bio2vec(sequence_dict,out_file_path,max_length)
    with open(dataset + '/max_length.txt','w') as f:
        print(max_length,file=f)
    print('embedding files at:%s; max_length=%s'%(out_file_path,max_length))



dataset = ['toy_cls','BindingDB_cls','DrugAI','CYP_cls'][int(sys.argv[1])]
embedding_type = ['onehot', 'bio2vec', 'tape'][int(sys.argv[2])]
# 'esm2' on colab
dataset_path = "../datasets/" + dataset + '/train/'
generate_embeddings("../datasets/" + dataset, embedding_type)
generate_fold(dataset_path)