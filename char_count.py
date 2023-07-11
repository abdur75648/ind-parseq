##############################################################################
# To Be Used For Including Train OccurencesOf Char In Char-Accuracy-CSV-File #
##############################################################################

import lmdb
import pandas as pd
from tqdm import tqdm

################## Getting Train Occurences From LMDB File ##################
data_dir = "../scratch/data/Telugu/LMDB/train/real/1M_realistic"
env = lmdb.open(data_dir, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print("Number of samples: ",nSamples)
def get_label(index):
    assert index < nSamples+1, 'index range error'
    with env.begin(write=False) as txn:
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode('utf-8')
    return label
lines = []
for i in range(nSamples+1):
    label = get_label(i)
    lines.append(label)

# print(len(lines))
# print(lines[:10])
# exit()
################## Getting Train Occurences From A Txt File ##################
# gt_file = "../scratch/data/Telugu/1M_synth/labels.txt"
# with open(gt_file,"r") as file:
#     lines = file.readlines()
# lines = [line.split()[1].strip() for line in lines]

train_occurence = {}
data = pd.read_csv("Character-wise-results.csv", header=0)
for i,row in data.iterrows():
    train_occurence[row["Alphabet"]]=0

for line in tqdm(lines):
    for char in line:
        if char in train_occurence.keys():
            train_occurence[char]+=1

train_occurence = {k: v for k, v in sorted(train_occurence.items(), key=lambda item: item[1],reverse=True)}

new_df = pd.DataFrame(columns=["Alphabet","Accuracy","Occurence","TotalOccurence"])
for i,row in data.iterrows():
    new_df = new_df.append({"Alphabet":row["Alphabet"],"Accuracy":row["Accuracy"],"Occurence":row["Occurence"],"TrainOccurence":train_occurence[row["Alphabet"]]},ignore_index=True)

new_df.to_csv("Char-report.csv",index=False)