from rdkit import Chem

import torch
import numpy as np
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import pandas as pd
from scipy.special import softmax
from sklearn import preprocessing
import csv
import torch_geometric as geo
'''
Drug_id = []
Drug_feature = []
with open('./data/drug/drug.csv', mode ='r')as file:
  csvFile = csv.DictReader(file)
  for lines in csvFile:
        Drug_id.append(lines['Drug'])
        Drug_feature.append(lines['SMILES'])

smiles_list = Drug_feature

# create a list of mols
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
#descriptors = list(np.array(Descriptors._descList)[:,0])
k = np.delete(np.array(Descriptors._descList)[:,0],[11,12,13,14,18,19,20,21,22,23,24,25],0)
k2 = np.delete(k,[57,70,150,162,163,185,194],0)
descriptors = list(k2)
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)
drug_feat = np.array([np.array(calculator.CalcDescriptors(mol)) for mol in mols])
print(drug_feat.shape)
print(np.where((~drug_feat.any(axis=0)))[0])
count = 0
nan_set = set()
index = np.argwhere(np.isnan(drug_feat))
for i in range(index.shape[0]):
    nan_set.add(index[i][1])
print(nan_set)
print(np.max(drug_feat))
print(np.min(drug_feat))
drug_feat = preprocessing.normalize(drug_feat)
print(np.max(drug_feat))
print(np.min(drug_feat))
np.save("./data/drug/drug_feat.npy",drug_feat)
'''

'''
df=pd.read_csv('./data/drug/tree-numbers.tsv',sep='\t')
drug_feat = np.load("./data/drug/drug_feat.npy")
#print(df["mesh_id"])

#print(df["mesh_tree_number"])
disease_id = []
with open('./data/drug/disease.csv', mode ='r')as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        disease_id.append(lines['Disease'])
#label_dict = {}
label_unique_id = set()
for dise_id in disease_id:
    for x in df.loc[df['mesh_id']==dise_id]["mesh_tree_number"]:
        label_unique_id.add(x[0:3])
print(len(label_unique_id))
print(label_unique_id)
print("disease node number"
label_list = sorted(list(label_unique_id))
Label_count = []
print(label_list)
for dise_id in disease_id:
    label_count = np.zeros(len(label_unique_id))
    for x in df.loc[df['mesh_id']==dise_id]["mesh_tree_number"]:
        label_count[label_list.index(x[0:3])]+=1
    Label_count.append(label_count)
Label_count = np.array(Label_count)
print(Label_count)
dise_drug = [[],[]]
with open('./data/drug/dise-drug.csv', mode ='r')as file:
    csvFile = csv.DictReader(file)
    for lines in csvFile:
        dise_drug[0].append(int(lines['Drug']))
        dise_drug[1].append(int(lines['Disease']))
dise_drug = torch.tensor(dise_drug,dtype=torch.long)
print(dise_drug)

Label_drug = np.zeros((drug_feat.shape[0],len(label_unique_id)))
for i in range(len(dise_drug[0])):
    Label_drug[dise_drug[0][i]]+=Label_count[dise_drug[1][i]]
print("unnormalized",Label_drug)
Label_drug_distribution = softmax(Label_drug,axis=1)
print("distribution",Label_drug_distribution)
print(np.sum(Label_drug_distribution[0]))
print(np.sum(np.max(Label_drug_distribution,axis=1)<0.7)/drug_feat.shape[0])
#np.save("./data/drug/labels.npy",Label_drug_distribution)
'''


def return_druggraph():
    hetero_graph = geo.data.HeteroData()
    hetero_graph['drug'].x = torch.from_numpy(np.load("./data/drug/drug_feat.npy")).float()
    hetero_graph['drug'].y = torch.from_numpy(np.load("./data/drug/labels.npy")).float()
    drug_dise = [[],[]]
    with open('./data/drug/dise-drug.csv', mode ='r')as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            drug_dise[0].append(int(lines['Drug']))
            drug_dise[1].append(int(lines['Disease']))
    dise_drug = torch.tensor([drug_dise[1],drug_dise[0]],dtype=torch.long)
    drug_dise = torch.tensor(drug_dise,dtype=torch.long)
    drug_prot = [[],[]]
    with open('./data/drug/drug-protein.csv', mode ='r')as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            drug_prot[0].append(int(lines['Drug']))
            drug_prot[1].append(int(lines['Protein']))
    prot_drug = torch.tensor([drug_prot[1],drug_prot[0]],dtype=torch.long)
    drug_prot = torch.tensor(drug_prot,dtype=torch.long)
    prot_gene = [[],[]]
    with open('./data/drug/protein-gene.csv', mode ='r')as file:
        csvFile = csv.DictReader(file)
        for lines in csvFile:
            prot_gene[0].append(int(lines['Protein']))
            prot_gene[1].append(int(lines['Gene']))
    gene_prot = torch.tensor([prot_gene[1],prot_gene[0]],dtype=torch.long)
    prot_gene = torch.tensor(prot_gene,dtype=torch.long)
    hetero_graph['drug','to','disease'].edge_index = drug_dise
    hetero_graph['disease','to','drug'].edge_index = dise_drug
    hetero_graph['drug','to','protein'].edge_index = drug_prot
    hetero_graph['protein','to','drug'].edge_index = prot_drug
    hetero_graph['gene','to','protein'].edge_index = gene_prot
    hetero_graph['protein','to','gene'].edge_index = prot_gene
    return hetero_graph
    
#hetero_graph = return_druggraph()
#print(hetero_graph)


