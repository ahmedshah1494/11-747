import os
CUDA_VISIBLE_DEVICES = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
import json
import codecs
import itertools
import numpy as np
import torch
import torch.utils.data as D
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tqdm import trange
from model import *

class SentenceDataset(D.Dataset):
    """docstring for SentenceDataset"""
    def __init__(self, path):
        super(SentenceDataset, self).__init__()     
        data = np.load(path)
        self.labels = torch.tensor(data['y'])
        self.sents = torch.tensor(data['x'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,i):
        return self.sents[i], self.labels[i]

def getPreds(outputs):
    if len(outputs.shape) == 3:
        outputs = torch.mean(outputs, 2)
        preds = torch.max(outputs, 1)[1]
    else:
        preds = torch.max(outputs, 1)[1]
    return preds

def get_accuracy(outputs, labels):
    preds = getPreds(outputs)
    diff = preds - labels
    misses = torch.numel(torch.nonzero(diff))
    total = torch.numel(labels)
    acc = 1 - (float(misses)/total)
    return acc      

def getConfMat(preds, labels, ntags):
    mat = np.zeros((ntags, ntags))
    print(len(preds), len(labels))
    for pred, true in zip(preds,labels):        
        mat[true,pred] += 1    
    return mat

def plotConfMat(cm, mapping):
    df_cm = pd.DataFrame(cm, index = [mapping[i] for i in range(cm.shape[0])],
                  columns =  [mapping[i] for i in range(cm.shape[0])])
    plt.figure(figsize = (20,20))
    heatmap = sn.heatmap(df_cm, annot=True)
    loc, labels = plt.xticks()
    heatmap.set_xticklabels(labels, rotation=45, fontdict={'fontsize':15}, ha="right")
    heatmap.set_yticklabels(labels, rotation=45, fontdict={'fontsize':15})
    return heatmap

def plotFromCSV(modelName, loss_cols, acc_cols):
    data = np.loadtxt('logs/'+modelName+'.csv',skiprows=1,delimiter=' ')
    epoch = data[:,[0]]
    acc = data[:,[acc_cols[0]]]
    loss = data[:,[loss_cols[0]]]
    val_acc = data[:,[acc_cols[1]]]
    val_loss = data[:,[loss_cols[1]]]

    fig, ax1 = plt.subplots()
    ax1.plot(acc)
    ax1.plot(val_acc)
    ax2 = ax1.twinx()
    ax2.plot(loss,color='r')
    ax2.plot(val_loss,color='g')
    plt.title('model loss & accuracy')
    ax1.set_ylabel('accuracy')
    ax2.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['training acc', 'validation acc'], loc=1)
    ax2.legend(['training loss', 'validation loss'], loc=7)
    fig.tight_layout()
    plt.savefig(modelName+'.png')
    plt.clf()    

def evaluate(model, val_loader, ntags, criterion):
    model = model.eval()
    total_val_acc = 0
    total_val_count = 0
    total_val_loss = 0

    pred_labels = [] 
    true_labels = []
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.long().to(device), y_batch.long().to(device)
        outputs = model.forward(x_batch)
        loss = criterion(outputs, y_batch)
        total_val_loss += float(loss)                  
        total_val_acc += get_accuracy(outputs, y_batch)

        pred_labels += list(getPreds(outputs).cpu().detach().numpy())
        true_labels += list(y_batch.cpu().detach().numpy())

        total_val_count += 1.0
    val_avg_metric = float(total_val_acc) / (total_val_count)
    val_avg_loss = total_val_loss/total_val_count
    print(val_avg_metric)
    confmat = getConfMat(pred_labels, true_labels, ntags)
    return val_avg_loss, val_avg_metric, confmat

def predict(model, label_mapping):
    model = model.eval()
    test_dataset = SentenceDataset(VAL_PATH)
    test_loader = D.DataLoader(test_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    pred_labels = []
    for x,_ in test_loader:
        outputs = model(x.to(device))
        if len(outputs.shape) == 3:
            outputs = torch.mean(outputs, 2)
            preds = torch.max(outputs, 1)[1]
        else:
            preds = torch.max(outputs, 1)[1]
        pred_labels += list(preds.cpu().detach().numpy())
    label_mapping = {label_mapping[k]:k for k in label_mapping.keys()}
    pred_labels = [label_mapping[i] for i in pred_labels]
    return pred_labels

def train(model):
    print("Loading data....")
    train_dataset = SentenceDataset(TRAIN_PATH)
    train_loader = D.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    val_dataset = SentenceDataset(VAL_PATH)
    val_loader = D.DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    n_train_batches = (len(train_dataset) + TRAIN_BATCH_SIZE - 1)//TRAIN_BATCH_SIZE
    n_val_batches = (len(val_dataset) + VAL_BATCH_SIZE-1)//VAL_BATCH_SIZE

    print(n_train_batches, n_val_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    # criterion = EnsembleXEntropy(L=0.001)

    max_metric = -1
    history = []
    for e in range(N_EPOCHS):
        total_loss = 0
        total_acc = 0
        total_eer = 0
        total_jsd = 0
        count = 1    
        fg = train_loader.__iter__()    
        with trange(n_train_batches) as t:
            for i in t:
                t.set_description("Epoch %d" % e)
                model.train()
                optimizer.zero_grad()
                x_batch, y_batch = fg.next()
                x_batch, y_batch = x_batch.long().to(device), y_batch.long().to(device)
                outputs = model.forward(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += float(loss)
                outputs = outputs.cpu().detach()
                y_batch = y_batch.cpu()
                
                total_acc += get_accuracy(outputs, y_batch)
                avg_loss = total_loss/count
                avg_acc = total_acc/count

                # total_jsd += float(criterion.avg_jsd)
                count += 1.0
                if i == n_train_batches-1 and val_loader != None:
                    total_val_acc = 0
                    total_val_count = 0
                    total_val_loss = 0
                    fg = val_loader.__iter__()
                    for val_s in range(n_val_batches):
                        model.eval()
                        x_batch, y_batch = fg.next()
                        x_batch, y_batch = x_batch.long().to(device), y_batch.long().to(device)
                        outputs = model.forward(x_batch)
                        loss = criterion(outputs, y_batch)

                        total_val_loss += float(loss)                  
                        total_val_acc += get_accuracy(outputs, y_batch)
                        total_val_count += 1.0
                    val_avg_metric = float(total_val_acc) / (total_val_count)
                    val_avg_loss = total_val_loss/total_val_count

                    scheduler.step(val_avg_loss)
                    t.set_postfix(loss=avg_loss, acc=avg_acc, jsd=total_jsd/total_val_count, val_loss=val_avg_loss, val_acc=val_avg_metric)
                    history.append([e, avg_loss, avg_acc, val_avg_loss, val_avg_metric])
                else:
                    t.set_postfix(loss=avg_loss, acc=avg_acc, jsd=total_jsd/count, val_loss=0, val_acc=0)
        if val_avg_metric > max_metric:
            max_metric = val_avg_metric
            model_name = 'multiC_model_2'
            torch.save(model.state_dict(), 'checkpoints/%s.pt'%model_name)
            np.savetxt('logs/%s.csv'%model_name, np.array(history), header='epoch train_loss train_acc val_loss val_acc')
            
if __name__ == '__main__':
    #++++++CONSTANTS+++++++
    TRAIN_PATH = 'topicclass/train.npz'
    VAL_PATH = 'topicclass/valid.npz'
    TEST_PATH = 'topicclass/test.npz'

    EMBED_PATH = 'w2v.npy'

    WORD_MAP_PATH = 'topicclass/word_mapping.json'
    LABEL_MAP_PATH = 'topicclass/label_mapping.json'

    SAVE_DIRECTORY = './checkpoints/'
    LOG_DIRECTORY = './logs/'

    USE_CUDA = True

    EMB_SIZE = 300
    FILTER_SIZE = 100
    WIN_SIZES = [3,4,5]

    TRAIN_BATCH_SIZE = 50
    VAL_BATCH_SIZE = 2048
    N_EPOCHS = 10

    UNK = 'UNK'
    PAD = '<p>'

    device =  torch.device("cuda" if USE_CUDA else "cpu")
    #++++++++++++++++++++++

    with codecs.open(WORD_MAP_PATH,'r','utf-8') as f:
        word_mapping = json.load(f)
    with codecs.open(LABEL_MAP_PATH,'r','utf-8') as f:
        label_mapping = json.load(f)


    print('setting up model...')
    nwords = len(word_mapping.keys())
    ntags = len(label_mapping.keys())


    w2v_embed = torch.from_numpy(np.load(EMBED_PATH))
    model = CNN(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZES, ntags, init_embed=w2v_embed)
    # filter_sizes = [64,128]
    # window_sizes = itertools.combinations([2,3,4,5], 3)
    # win_filt_params = [x for x in itertools.product(filter_sizes,window_sizes)]
    # print (win_filt_params)
    # model = Ensemble([CNN(nwords, EMB_SIZE, fs, ws, ntags, init_embed=w2v_embed) for (fs, ws) in win_filt_params], ntags)
    # model.to(device)

    # train(model)

    model.load_state_dict(torch.load('checkpoints/multiC_model_2.pt'))    
    model.to(device)

    preds = predict(model, label_mapping)
    np.savetxt('val_predictions.txt', preds, fmt='%s')

    # val_dataset = SentenceDataset(VAL_PATH)
    # val_loader = D.DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)
    # loss, metric, confmat = evaluate(model, val_loader, len(label_mapping.keys()), nn.CrossEntropyLoss())    
    # np.savetxt('dev_confmat.txt', confmat)
    
    # confmat = np.loadtxt('dev_confmat.txt')
    # plotConfMat(confmat, {label_mapping[k]:k for k in label_mapping.keys()}).figure.savefig('dev_confmat.png')
    
    # plotFromCSV('multiC_model_2', [1,3],[2,4])