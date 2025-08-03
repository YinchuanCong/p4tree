import torch 
import numpy as np   
import pandas as pd  
from sklearn.model_selection import train_test_split
import os, sys
from torch.autograd import Variable
import torch.optim as optim
import math 
from torch import nn                     
from torch.utils.data import DataLoader,TensorDataset,Dataset
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from functional import reset_normal_param, LinearWeightNorm
from itertools import cycle
from sklearn.metrics import f1_score,classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import time 


from sklearn.tree import DecisionTreeClassifier
from decistirontree import SoftLabelDecisionTree
from decisionTree.SoftTree import SoftTreeClassifier

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)    
# from utils.util import TabularDataset,SimpleMLP,LSTM,GRU
# from utils.util import update_ema,pseudo_labeling,select_features 
import tensorboardX



class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]



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


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim = 28 ** 2):
        super(Generator, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.z_dim = z_dim
        self.fc1 = nn.Linear(z_dim, 500, bias = False)
        self.bn1 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.fc2 = nn.Linear(500, 500, bias = False)
        self.bn2 = nn.BatchNorm1d(500, affine = False, eps=1e-6, momentum = 0.5)
        self.fc3 = LinearWeightNorm(500, output_dim, weight_scale = 1)
        self.bn1_b = Parameter(torch.zeros(500))
        self.bn2_b = Parameter(torch.zeros(500))
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, batch_size):
        x = Variable(torch.rand(batch_size, self.z_dim), requires_grad = False, volatile = not self.training)
        x= x.to(self.device)
        x = F.softplus(self.bn1(self.fc1(x)) + self.bn1_b)
        x = F.softplus(self.bn2(self.fc2(x)) + self.bn2_b)
        x = F.softplus(self.fc3(x))
        return x


class SSGAN:
    def __init__(self,input_csv, G_model, D_model , labeled_ratio=0.1, test_ratio=0.2,output_class = 8):
        pass
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.X_labeled, self.y_labeled,self.X_unlabeled,self.X_test,self.y_test = self._split_dataset(input_csv,labeled_ratio,test_ratio) 
        
        input_dim = self.X_labeled.shape[1]
        num_classes = len(np.unique(self.y_labeled))
        self.G = G_model(input_dim,input_dim).to(self.device)
        self.D = D_model(input_dim, output_class).to(self.device) # 判断是否是伪造的数据
        self.Doptim = optim.Adam(self.D.parameters(), lr = 1e-3, betas = (0.95,0.9999))
        self.Goptim = optim.Adam(self.G.parameters(), lr = 1e-3, betas = (0.95,0.9999)) 
        
        self.writer = tensorboardX.SummaryWriter(log_dir = 'log.txt')
                
    def _split_dataset(self, input_csv, labeled_ratio=0.1, test_ratio=0.2, seed=42 ):
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
        "UDPdstport",'label',])#'PktNumber'
    
        # df = df[:1000000] # for debug
        df = shuffle(df, random_state=42)
        df = df[:1000000] # for debug
        
        features = df.drop(columns=['FlowKey','TimestampNano','label',])#'PktNumber'
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
    
    def _log_sum_exp(self,x,axis = 1):
        m = torch.max(x,dim=1)[0]
        return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))
    
    def _reset_normal_param(L, stdv, weight_scale = 1.):
        assert type(L) == torch.nn.Linear
        torch.nn.init.normal(L.weight, std=weight_scale / math.sqrt(L.weight.size()[0]))
    
    def trainD(self,x,y,unlabel, w_unlabeled=0.8):
        x_label, x_unlabel, y = Variable(x).to(self.device), Variable(unlabel).to(self.device), Variable(y, requires_grad = False).to(self.device)
        
        output_label, output_unlabel, output_fake = self.D(x_label),self.D(x_unlabel), self.D(self.G(x_unlabel.shape[0]).detach())
        
        logz_label, logz_unlabel, logz_fake = self._log_sum_exp(output_label),self._log_sum_exp(output_unlabel), self._log_sum_exp(output_fake)
        prob_label = torch.gather(output_label, 1, y.unsqueeze(1)) # log e^x_label = x_label 
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
        loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
        
        loss = loss_supervised + w_unlabeled * loss_unsupervised
        acc = torch.mean((output_label.max(1)[1] == y).float())
        self.Doptim.zero_grad()
        loss.backward()
        self.Doptim.step()
        return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc
        
    
    def trainG(self, x_unlabel):
        fake = self.G(x_unlabel.size()[0]).view(x_unlabel.size())
        mom_gen, output_fake = self.D(fake, feature=True)
        mom_unlabel, _ = self.D(Variable(x_unlabel), feature=True)
        mom_gen = torch.mean(mom_gen, dim = 0)
        mom_unlabel = torch.mean(mom_unlabel, dim = 0)
        loss_fm = torch.mean((mom_gen - mom_unlabel) ** 2)
        loss = loss_fm 
        self.Goptim.zero_grad()
        self.Doptim.zero_grad()
        loss.backward()
        self.Goptim.step()
        return loss.data.cpu().numpy()

    def _nonsemi(self,epochs):
        for epoch in range(epochs):
            self.D.train()
            labeled_dataset = TabularDataset(self.X_labeled,self.y_labeled)
            label_loader = cycle(DataLoader(labeled_dataset, batch_size = 1024, shuffle=True, drop_last=True))
                                
    def train(self,epochs):
        unlabeled_dataset = TabularDataset(self.X_unlabeled)
        unlabeled_loader1 = DataLoader(unlabeled_dataset,batch_size = 1024, shuffle=True, drop_last=False)
        unlabel_loader2 = cycle(DataLoader(unlabeled_dataset, batch_size = 1024, shuffle=True, drop_last=True))
        
        labeled_dataset = TabularDataset(self.X_labeled,self.y_labeled)
        label_loader = cycle(DataLoader(labeled_dataset, batch_size = 1024, shuffle=True, drop_last=True))
        for epoch in range(epochs):
            self.G.train()
            self.D.train()
            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0
            batch_num = 0
            for _, unlabel1 in enumerate(unlabeled_loader1):
                batch_num+=1 
                
                unlabel2 = next(unlabel_loader2)
                x, y = next(label_loader)
                
                x, y, unlabel1, unlabel2 = x.to(self.device), y.to(self.device), unlabel1.to(self.device), unlabel2.to(self.device)
                l_supervised, l_unsupervised, acc = self.trainD(x,y,unlabel1)
                loss_supervised += l_supervised
                loss_unsupervised += l_unsupervised
                accuracy += acc 
                
                lg = self.trainG(unlabel2)
                
                if epoch > 1 and lg>1:
                    lg = self.trainG(unlabel2)
                
                loss_gen += lg 
                
                
                
                if batch_num+ 1 % 1000 == 0:
                    print('Training: %d / %d' % (batch_num + 1, len(unlabeled_loader1)))
                
                    self.writer.add_scalars('loss', {'loss_supervised':l_supervised, 'loss_unsupervised':l_unsupervised, 'loss_gen':lg},lg)
                    
                    self.D.train()
                    self.G.train()
                    
            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))
            accuracy /= batch_num
            sys.stdout.flush()
            path = "./pkls/"+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        torch.save(self.G,path+"_Generator.pkl")
        torch.save(self.D,path+"_Discrinator.pkl")
            
    
    def classification(self,x):
        with torch.no_grad():
            ret = torch.max(self.D(Variable(x)), 1)[1].data
        return ret     
    
    def eval(self,G_pkl=None, D_pkl=None):
        if G_pkl !=None and D_pkl!=None:
            self.G=torch.load(G_pkl)
            self.D=torch.load(D_pkl)
        
        self.G.eval()
        self.D.eval()
        
        d,l = [],[]
        
        # for (datnum,label) in zip(self.X_test.values.tolist(),self.y_test.values.tolist()):
        #     d.append(datnum)               
        #     l.append(label)
        
        x, y = torch.Tensor(self.X_test.values.tolist()).to(self.device), torch.LongTensor(self.y_test.values.tolist()).to(self.device)
        pred = self.classification(x)
        print(f1_score(y.cpu().data.tolist(),pred.cpu().data.tolist(),average='weighted'))
        print(classification_report(y.cpu().data.tolist(),pred.cpu().data.tolist()))
        return torch.sum(pred == y)
    
    # 获得伪标签
    def get_pesolabel(self,G_pkl=None, D_pkl=None):
        pass 
        if G_pkl !=None and D_pkl!=None:
            self.G=torch.load(G_pkl)
            self.D=torch.load(D_pkl)
        self.G.eval()
        self.D.eval()
        
        d,l = [],[]
        
        for (datnum,label) in zip(self.X_test,self.y_test):
            d.append(datnum)               
            l.append(label)
            
        X_tensor = torch.tensor(self.X_unlabeled.values,dtype=torch.float32).to(self.device)
        # peso_y = self.D(X_tensor)
        unlabeled_loader1 = DataLoader(X_tensor,batch_size = 1024, shuffle=True, drop_last=False)
        pesos=[]
        with torch.no_grad():
            for _,x in enumerate(unlabeled_loader1):
                peso = self.D(x)
                pesos.append(peso.cpu())
        
        peso_y = torch.concat(pesos,dim=0)
        peso_y = F.softmax(peso_y,dim=1)
        return peso_y
        
        
        # X_combined = pd.concat([self.X_labeled,self.X_labeled],ignore_index=True)
        
        # x, y = torch.stack(d).to(self.device), torch.LongTensor(l).to(self.device)
        # pred = self.classification(x)
        # print(f1_score(y,pred,average='weighted'))
        # print(classification_report(y,pred))
    
    def C45TreeBuild(self):
        peso_y = self.get_pesolabel("pkls/2025-07-30 00:33:13_Generator.pkl","pkls/2025-07-30 00:33:13_Discrinator.pkl")
        X_combined = pd.concat([self.X_labeled,self.X_unlabeled],ignore_index=True)
        
        y_labeled = F.one_hot(torch.tensor(self.y_labeled.tolist())).tolist()
        # y_combined = pd.concat([pd.Series(y_labeled),pd.Series(peso_y.tolist())],ignore_index=True)
        y_combined = np.array(y_labeled+peso_y.tolist())
        
        
        
        # X_expanded, y_expanded, w_expanded = [], [], []
        # for xi, yi in zip(X_combined.values.tolist(), y_combined):
        #     for cls, prob in enumerate(yi):
        #         if prob > 0:  # 跳过0概率的类别
        #             X_expanded.append(xi)
        #             y_expanded.append(cls)
        #             w_expanded.append(prob)

        # X_expanded = np.array(X_expanded)
        # y_expanded = np.array(y_expanded)
        # w_expanded = np.array(w_expanded)
        # clf = DecisionTreeClassifier(criterion="log_loss",min_samples_split=128, random_state=0,max_features='sqrt')
        # clf.fit(X_expanded,y_expanded,sample_weight=w_expanded)

        
        
        clf = SoftTreeClassifier(8,n_features='sqrt', min_sample_leaf=20)
        
        
        clf.fit(X_combined.to_numpy(), y_combined,features_attr=['d' for i in range(len(X_combined.to_numpy().tolist()[0])-1)])
        y_pred = clf.predict(self.X_test.to_numpy())
        # y_pred = np.argmax(y_pred,axis=1)
        print("📊 C4.5 Classification Report:")
        print(classification_report(self.y_test.to_numpy(), y_pred))
        
    def TreeBuild_simpleWeight(self):
        peso_y = self.get_pesolabel("pkls/2025-07-30 00:33:13_Generator.pkl","pkls/2025-07-30 00:33:13_Discrinator.pkl")
        X_combined = pd.concat([self.X_labeled,self.X_unlabeled],ignore_index=True)
        
        y_labeled = F.one_hot(torch.tensor(self.y_labeled.tolist())).tolist()
        # y_combined = pd.concat([pd.Series(y_labeled),pd.Series(peso_y.tolist())],ignore_index=True)
        y_combined = np.array(y_labeled+peso_y.tolist())
        X_expanded, y_expanded, w_expanded = [], [], []
        for xi, yi in zip(X_combined.values.tolist(), y_combined):
            for cls, prob in enumerate(yi):
                if prob > 0:  # 跳过0概率的类别
                    X_expanded.append(xi)
                    y_expanded.append(cls)
                    w_expanded.append(prob)

        X_expanded = np.array(X_expanded)
        y_expanded = np.array(y_expanded)
        w_expanded = np.array(w_expanded)
        clf = DecisionTreeClassifier(criterion="log_loss",min_samples_split=128, random_state=0,max_features='sqrt')
        clf.fit(X_expanded,y_expanded,sample_weight=w_expanded)
        # clf.fit(X_combined.to_numpy(), y_combined)
        y_pred = clf.predict(self.X_test.to_numpy())
        # y_pred = np.argmax(y_pred,axis=1)
        print("📊 C4.5 Classification Report:")
        print(classification_report(self.y_test.to_numpy(), y_pred))
    


if __name__ == '__main__':
    pass          
    seed = 42 
    ssgan = SSGAN('/sdc1/labeled_datasets/ISCX_tornotor_application_pktlevel.csv',Generator,Discriminator,labeled_ratio=0.15)
    ssgan.train(100)
    ssgan.eval()
    # ssgan.C45TreeBuild()
    # ssgan.TreeBuild_simpleWeight()
    