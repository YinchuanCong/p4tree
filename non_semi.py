
import torch 
import numpy as np   
import pandas as pd  
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,f1_score
from sklearn.tree import export_text         
from sklearn.feature_selection import RFE
import itertools
import os  
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)                   

from SSGAN import TabularDataset,Discriminator
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn.utils import shuffle



class LinearWeightNorm(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_scale=None, weight_init_stdv=0.1):
        super(LinearWeightNorm, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(out_features, in_features) * weight_init_stdv)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if weight_scale is not None:
            assert type(weight_scale) == int
            self.weight_scale = Parameter(torch.ones(out_features, 1) * weight_scale)
        else:
            self.weight_scale = 1 
    def forward(self, x):
        W = self.weight * self.weight_scale / torch.sqrt(torch.sum(self.weight ** 2, dim = 1, keepdim = True))
        return F.linear(x, W, self.bias)
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', weight_scale=' + str(self.weight_scale) + ')'


class Discriminator(nn.Module):
    def __init__(self, input_dim = 28 ** 2, output_dim = 10):
        super(Discriminator, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.input_dim = input_dim
        self.layers = torch.nn.ModuleList([
            LinearWeightNorm(input_dim, 1000),
            LinearWeightNorm(1000, 500),
            LinearWeightNorm(500, 250),
            LinearWeightNorm(250, 250),
            LinearWeightNorm(250, 250)]
        )
        self.final = LinearWeightNorm(250, output_dim, weight_scale=1)

    def forward(self, x, feature = False):
        x = x.view(-1, self.input_dim)
        noise = torch.randn(x.size()) * 0.3 if self.training else torch.Tensor([0])
        noise = noise.to(self.device)
        x = x + Variable(noise, requires_grad = False)
        for i in range(len(self.layers)):
            m = self.layers[i]
            x_f = F.relu(m(x))
            noise = torch.randn(x_f.size()) * 0.5 if self.training else torch.Tensor([0])
            
            x = (x_f + Variable(noise, requires_grad = False).to(self.device))
        if feature:
            return x_f, self.final(x)
        return self.final(x)

class DNNClassifier(nn.Module):
    def __init__(self,input_csv,model=Discriminator,labeled_ratio=0.1, test_ratio=0.2) -> None:
        super().__init__()
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.X_labeled, self.y_labeled,self.X_unlabeled,self.X_test,self.y_test = self._split_dataset(input_csv,labeled_ratio,test_ratio) 
        
        input_dim = self.X_labeled.shape[1]
        num_classes = len(np.unique(self.y_labeled))
        self.model= model(input_dim, num_classes).to(device)
        self.device = device 
        
        
        
    def _split_dataset(self, input_csv, labeled_ratio=0.5, test_ratio=0.2, seed=42):
        df = pd.read_csv(input_csv,header=1, names = ["FlowKey",
		"TimestampNano",
		"PacketSize",
		"EtherType",
		"IPv4Protocol",
		"IPv4Flags",
		"IPv6Next",
		"IPv6Options",
		"TCPsrcport",
		"TCPdstport",
		"Tcpflags",
		"UDPsrcport",
		"UDPdstport",'label'])
    
        # df = df[:1000000] # for debug
        df = shuffle(df, random_state=42)
        df = df[:1000000] # for debug
        

        features = df.drop(columns=['FlowKey','TimestampNano','label'])
        # labels = df['label']
        labels,uniques = pd.factorize(df['label'])
        labels = pd.Series(labels)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_ratio, stratify=labels, random_state=seed
        )
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X_temp, y_temp, test_size=1 - labeled_ratio / (1 - test_ratio),
            stratify=y_temp, random_state=seed
        )
        return X_labeled, y_labeled, X_unlabeled, X_test, y_test 

    def train_classifer(self,loader,optimizer,criterion, epochs=100,lr=1e-3):
        self.model.train()
        device = self.device 
        
        preds, grdtruth = None,None 
        for epoch in range(epochs):
            for X,label in loader:
                X, y = X.to(device), F.one_hot(label,num_classes=8).float().to(device)
                out = self.model(X)
                
                loss = criterion(out,y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if preds ==None :
                    pred = out 
                    grdtruth = y 
                else:
                    pred = torch.concat([pred,out],dim=1)
                    grdtruth = torch.concat([grdtruth,y],dim=1)
                    
            if epoch % 100 == 0:
                with torch.no_grad():
                    loss = criterion(pred,grdtruth)
                    pred = pred.argmax(dim=1).data.cpu().tolist()
                    grdtruth = grdtruth.argmax(dim=1).data.cpu().tolist()
                    train_f1 = f1_score(grdtruth,pred,average="weighted")
                    print(f"Training epoch {epoch}: train data f1: {train_f1:.4f}")
            pred,grdtruth = None,None 
            
                
    def NN_classifier(self,lr,epochs):
        device = self.device 
        train_ds = TabularDataset(self.X_labeled, self.y_labeled)
        train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
                
        self.train_classifer(train_loader,optimizer,criterion,epochs,lr)
        
        teacher_outs = 0
        student_outs =0 
        train_grdtruth = 0 
        for X,y in train_loader:
            with torch.no_grad():
                X,y = X.to(device),y.to(device)
                student_out = self.model(X).argmax(dim=1)
                if type(teacher_outs) == int :
                    student_outs = student_out
                    train_grdtruth = y
                else:
                    # teacher_outs = torch.concat([teacher_outs,teacher_out],0)
                    student_outs = torch.concat([student_outs,student_out],0)
                    train_grdtruth = torch.concat([train_grdtruth,y],0)
        
        # teacher_outs = torch.unsqueeze(teacher_outs,0)
        # student_outs = torch.unsqueeze(student_outs,0)
        # train_grdtruth = torch.unsqueeze(train_grdtruth,0)
        # train_teacher_f1 = f1_score(train_grdtruth.data.cpu(),teacher_outs.data.cpu())
        train_student_f1 = f1_score(train_grdtruth.data.cpu(),student_outs.data.cpu(),average='weighted')
        print(f"Finish training! \n on training data,  model  f1: %f \n"%(train_student_f1))
        
        
        grdtruth = None
        pred = None 
        test_ds = TabularDataset(self.X_test,self.y_test)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        with torch.no_grad():
            for X,y in test_loader:
                X= X.to(device)
                y = y.to(device)
                
                out = self.model(X).argmax(dim=1)
                if pred == None :
                    pred = out 
                    grdtruth = y 
                else :
                    pred = torch.concat([pred,out],dim=0)
                    grdtruth = torch.concat([grdtruth,y],dim=0)
                
            pred = pred.squeeze().data.cpu().tolist()
            grdtruth = grdtruth.squeeze().data.cpu().tolist()
        print(f1_score(grdtruth,pred,average='weighted'))
        print(classification_report(grdtruth,pred))


if __name__ =='__main__':
    mlpclassifier = DNNClassifier('/sdc1/labeled_datasets/ISCX_tornotor_application_pktlevel.csv',model=Discriminator,labeled_ratio=0.15)
    mlpclassifier.NN_classifier(1e-3,101)
                
            
            
            
            
