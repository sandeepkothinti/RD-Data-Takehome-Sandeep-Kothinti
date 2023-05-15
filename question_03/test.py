from dataset import myDS
from model import simpleModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from PIL import Image
import torch

sizes = [64,64]
ds1 = myDS('../question_01/full_data.csv', mode='train', out_size=sizes)
ds2 = myDS('../question_01/full_data.csv', mode='valid', out_size=sizes)
ds3 = myDS('../question_01/full_data.csv', mode='test', out_size=sizes)

bce_loss = torch.nn.BCELoss()
DL_train = DataLoader(ds1, batch_size=64, shuffle=True)
DL_valid = DataLoader(ds2, batch_size=64)
DL_test = DataLoader(ds3, batch_size=128)
model1 = simpleModel(in_size=sizes[0])
optim = torch.optim.Adam(model1.parameters(), lr=5e-3)

prev_valid_loss = 1e9 
for epoch in range(20):
    train_loss =0 
    valid_loss = 0
    for data, labels in tqdm(DL_train):
        optim.zero_grad()
        output = model1(torch.unsqueeze(data,1))
        loss = bce_loss(output, torch.unsqueeze(labels, 1))
        loss.backward()
        optim.step()
        train_loss += loss.item()
    
    for data, labels in tqdm(DL_valid):
        with torch.no_grad():
            output = model1(torch.unsqueeze(data,1))
            loss = bce_loss(output, torch.unsqueeze(labels, 1))
            valid_loss += loss.item()
    if valid_loss < prev_valid_loss:
        torch.save(model1.state_dict(), 'best_model')
    print('Epoch ', epoch, ' train losses ', train_loss, ' valid losses ', valid_loss )

# testing the model on test set
model1.load_state_dict(torch.load('best_model'))
for data, labels in DL_test:
    output = model1(torch.unsqueeze(data,1)).detach().numpy()
    output_th = np.where(output>0.5, 1, 0)
    print('test accuracy ' , accuracy_score(output_th, labels.numpy()))
    print('test roc ', roc_auc_score(labels.numpy(), output))


# testing on the given dataset
test_path = '../rd_test_dataset/'
test_files = os.listdir(test_path)
outfile = open('rd_output.csv', 'w')
outfile.writelines('filename,label\n')
for file in test_files:
    if file.startswith('.'):
        continue
    image = Image.open(test_path+file)
    image = np.asarray(image.convert('L').resize(sizes), dtype=np.float32)
    output = model1(torch.unsqueeze(torch.Tensor(image),0)).detach().numpy()
    if output>0.5:
        outfile.writelines(file+','+'fake\n')
    else:
        outfile.writelines(file+','+'real\n')


