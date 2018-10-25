import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.font_manager as font_manager
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics
 
 
#Define hyper-parameters
epoch = 2000
neurons = 50
learning_rate = 0.02
 
#Open file
file = 'data.xlsx'
xl = pd.ExcelFile(file)
 
#Get training data
df = xl.parse('train')
 
#Correlation checking with seaborn
#sb.pairplot(df,x_vars=['E','T','L','t','Conf','pH'], y_vars='X', kind='reg')
#print (df.corr())
 
#Data pre-processing
x_data = df[['E','T','L','t','Conf','pH']]
y_data = df[['X']]
 
#import pdb; pdb.set_trace()
 
x_train_np = x_data.values ############
y_train_np = y_data.values
#scaler = MinMaxScaler()
#x_train_np = scaler.fit_transform(x_train_np)
#y_train_np = scaler.fit_transform(y_train_np)
 
#Data to Tensor
x_train = Variable(torch.Tensor(x_train_np),requires_grad=False)#######
y_train = Variable(torch.Tensor(y_train_np),requires_grad=False)
 
#NN Architecture
class Model(torch.nn.Module):
 
    def __init__(self):
        """
        In the constructor, instantiate two nn.Linear module and one activation
        """
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(len(x_train[0]), neurons)  # One in and one out
        self.l2 = torch.nn.Linear(neurons, len(y_train[0]))
        self.Tanh = torch.nn.Tanh()
         
    def forward(self, x):
        """
        In the forward function, accept a Variable of input data and return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out = self.Tanh(self.l1(x))
        y_pred = self.l2(out)
        return y_pred
 
ANN = Model()
 
#Loading pre-saved model:
"""
Open a pre-saved model(if required). 
Jump to data post-processing step if pre-saved model is loaded
"""
#ANN.load_state_dict(torch.load('mytraining.pt'))
#ANN.eval()
#y_pred = ANN(x_train)
 
 
#Criteration and Optimizer
""" 
Construct loss function and an Optimizer. The call to model.parameters()
in the constructor will contain the learnable parameters of the two
nn.Linear modules which are members of the model.
"""
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ANN.parameters(), lr=learning_rate)
 
# Training loop
history_obj = []
y_Target    = []
for i in range(epoch):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = ANN(x_train)
 
    # Compute and print loss
    loss = criterion(y_pred, y_train)
    if i % (epoch/10) == 0:
        print(i, loss.item())
 
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # save result to the text
 
    history_obj.append(loss)
 
np.savetxt("./result_loss.txt", history_obj)
 
 
y_train1 = y_train.detach().numpy()
y_pred1 = y_pred.detach().numpy()
 
y_Target=np.hstack((y_train1,y_pred1))
 
#import pdb; pdb.set_trace()
np.savetxt("./result_y_target.txt", y_Target)
 
 
 
 
#Data post-processing
#Chart function for data visualization
def chart(Title,xymin,xymax):
 
    chart.fig = plt.figure(figsize=(5,5))
    chart.ax = chart.fig.add_subplot(111)
     
    for axis in ['top','bottom','left','right']:
        chart.ax.spines[axis].set_linewidth(2)
    chart.ax.axis([xymin, xymax, xymin, xymax])
    chart.ax.set_title(Title, fontname='Arial', fontsize=24)
    chart.ax.set_xlabel('True values', fontname='Arial', fontsize=20)
    chart.ax.set_ylabel('Predictions', fontname='Arial', fontsize=20)
    plt.xticks(fontname='Arial', fontsize = 14)
    plt.yticks(fontname='Arial', fontsize = 14)   
    chart.ax.tick_params(axis='both', length=7, width = 2)
     
    lims = [np.min([chart.ax.get_xlim(), chart.ax.get_ylim()]), # min of both axes 
            np.max([chart.ax.get_xlim(), chart.ax.get_ylim()]),] # max of both axes  
    chart.ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0,linewidth=2)
    chart.ax.set_aspect('equal')
    chart.ax.set_xlim(lims)
    chart.ax.set_ylim(lims)
     
    return
 
y_pred_np = y_pred.detach().numpy()
 
chart(Title='Training',xymin=0,xymax=6)
chart.ax.scatter(y_train_np,y_pred_np,s=100,alpha=0.75)
plt.show()
chart.fig.savefig('Training.jpeg', dpi=200)   
 
print('RMSE_train=', np.sqrt(metrics.mean_squared_error(y_train_np, y_pred_np)), 'R2_train=', metrics.r2_score(y_train_np, y_pred_np))
 
#Save parameters to csv
file_para = open('parameters.txt', 'w')
#file_para = open('parameters.csv', 'a')  
list_para = list(ANN.named_parameters())
#file_para.write(str(list_para))
 
#import pdb; pdb.set_trace()
 
 
for name, param in ANN.named_parameters():
#    if param.requires_grad:
    file_para = open('parameters.txt', 'a')  
    file_para.write('{}, {}\n'.format(name, param))
    print (name, param.data)
file_para.close()