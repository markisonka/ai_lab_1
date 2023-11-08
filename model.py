import numpy as np
import random
import matplotlib.pyplot as plt 

class Linear:
  def __init__(self,nin,nout):
    self.W = np.random.normal(0,1.0/np.sqrt(nin),(nout,nin))
    self.b = np.zeros((1,nout))

  def forward(self,x):
    self.x = x
    return np.dot(x,self.W.T)+self.b

  def backward(self,dz):
    self.dW = np.dot(dz.T,self.x)
    self.db = dz.sum(axis=0)
    return np.dot(dz,self.W)

  def update(self,lr):
    self.W -= lr*self.dW
    self.b -= lr*self.db  

class RelU:
    def forward(self,x):
        self.x = x
        x[x < 0] = 0.0
        return x
    def backward(self,dx):
      dx[self.x < 0] = 0.0
      return dx
    
# class Sigmoid:
#   def forward(self,x):
#     return 1/(np.exp(-x)+1)
#   def backward (self,dx):
#     expdx=np.exp(-dx)
#     return expdx/((expdx+1)*(expdx+1))

class PRelU:
    def forward(self,x,init=0.25):
        self.x = x
        self.init=init
        nx=np.where(x < 0,x*init,x)
        return nx
    def backward(self,dx):
      ndx=np.where(self.x < 0,dx*self.init,dx)
      return ndx
    
class ResLinear:
  def __init__(self,nin,centern, act = "RelU"):
    self.Linear1 = Linear(nin,centern)
    self.Linear2 = Linear(centern,nin)
    if act == "RelU":
      self.act = RelU()
    elif act == "Softmax":
      self.act = Softmax()
  def forward(self,x):
    self.x = x
    x1 = self.Linear1.forward(x)
    x1 = self.act.forward(x1)
    x1 = self.Linear2.forward(x1)
    return x+x1

  def backward(self,dz):
    dz1 = self.Linear2.backward(dz)
    dz1 = self.Linear1.backward(dz1)
    return dz + dz1

  def update(self,lr):
    self.Linear1.update(lr)
    self.Linear2.update(lr)

class Softmax:
  def forward(self,z):
    self.z = z
    zmax = z.max(axis=1,keepdims=True)
    expz = np.exp(z-zmax)
    Z = expz.sum(axis=1,keepdims=True)
    return expz/Z

  def backward(self,dp):
    p = self.forward(self.z)
    pdp = p * dp
    return pdp - p * pdp.sum(axis=1, keepdims=True)
       
class Net:
  def __init__(self,layers):
    self.layers = layers

  def forward(self,x):
    for l in self.layers:
      x = l.forward(x)
    return x

  def backward(self,x):
    for l in self.layers[::-1]:
      x = l.backward(x)
    return x

  def update(self,lr):
    for l in self.layers:
      if 'update' in l.__dir__():
        l.update(lr)

class CrossEntropyLoss:
  def forward(self,p,y):
    self.p = p
    self.y = y
    p_of_y = p[np.arange(len(y)), y]
    log_prob = -np.log(p_of_y)
    return log_prob.mean()

  def backward(self,loss):
    dlog_softmax = np.zeros_like(self.p)
    dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
    return dlog_softmax / self.p

class MSELoss:
  def forward(self,p,y):
    self.p = p
    self.y = y
    return np.sum((p.T-y)**2)
  
  def backward(self,loss):
    return 2*(self.p.T-self.y).T
 
class SGD:
  def __init__(self) -> None:
    pass
  def step(self, net, criterion, loss, lr):
    # backward pass
    dp = criterion.backward(loss)
    net.backward(dp)

    # update weights
    net.update(lr)

class Momentum_SGD:
  def __init__(self, gamma = 0.9) -> None:
    self.gamma = gamma
    self.dloss = 0

  def step(self, net, criterion, loss, lr):

    # backward pass
    dloss = criterion.backward(loss)
    self.dloss = self.gamma * self.dloss + lr * dloss
    net.backward(self.dloss)

    # update weights
    net.update(lr)

class Adam:
  def __init__(self, beta1 = 0.9,beta2=0.99,eps=1e-08) -> None:
    self.beta1 = beta1
    self.beta2 = beta2
    self.massif_Moment=0
    self.massif_Glad=0
    self.t=1
    self.eps=eps

  def step(self, net, criterion, loss, lr):
    # backward pass

    dp = criterion.backward(loss)
    # расчет шага
    self.massif_Moment=self.massif_Moment*self.beta1+(1-self.beta1)*dp
    self.massif_Glad=self.massif_Glad*self.beta1+(1-self.beta1)*(dp**2)
    massif_Moment_corected=self.massif_Moment/(1-np.power(self.beta1,self.t))
    massif_Glad_corected=self.massif_Glad/(1-np.power(self.beta2,self.t))
    exdp=massif_Moment_corected/np.sqrt(massif_Glad_corected + self.eps)

    net.backward(exdp)

    # update weights
    net.update(lr)
    self.t+=1
    
class RMSprop:
  def __init__(self, beta1 = 0.9,eps=1e-08) -> None:
    self.beta1 = beta1
    self.massif_Glad=0
    self.eps=eps

  def step(self, net, criterion, loss, lr):
    # backward pass

    dp = criterion.backward(loss)
    # расчет шага
    self.massif_Glad=self.massif_Glad*self.beta1+(1-self.beta1)* np.power(dp, 2)
    exdp=dp/np.sqrt(self.massif_Glad + self.eps)

    net.backward(exdp)

    # update weights
    net.update(lr)

def accuracy(net,x,y):
  z = net.forward(x)
  pred = np.argmax(z,axis=1)
  return (pred==y).mean()

def Loss(net, Loss, x, y):
  z = net.forward(x)
  return Loss.forward(z,y)


def set_seed(seed = 10):
        '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
        np.random.seed(seed)
        random_state = np.random.RandomState(seed)
        random.seed(seed)
        return random_state

random_state = set_seed(99)

def my_train_val_split(Data:list,Relationship=0.8,seed=None):
  TrainData=[]
  ValData=[]
  Shape=[]
  for inm in range(len(Data)):
    Shape.append([ Data[inm].shape[i] for i in range(len(Data[inm].shape))])
    Shape[inm][0]=-1
  if seed != None:
    set_seed(seed)
  for index in range(int(Data[0].shape[0])):
    if random.random()>=Relationship:
      if len(ValData) !=len(Data):
        for i in range(len(Data)):
          ValData.append(np.array(Data[i][index]))
      else:
        for i in range(len(Data)):
          ValData[i]=np.append(ValData[i],Data[i][index])
          ValData[i]=np.reshape(ValData[i],Shape[i])
    else:
      if len(TrainData) !=len(Data):
        for i in range(len(Data)):
          TrainData.append(np.array(Data[i][index]))
      else:
        for i in range(len(Data)):
          TrainData[i]=np.append(TrainData[i],Data[i][index])
          TrainData[i]=np.reshape(TrainData[i],Shape[i])
  return TrainData, ValData


class Module:
  def __init__(self, net, criterion, optim = SGD(), data = None, val_split = 0.2, data_train = None, data_val = None,seed=None,drop_last=True):
    self.net = net
    self.criterion = criterion
    self.optim = optim
    self.drop_last=drop_last
    if data_train and data_val:
      self.train_x, self.train_labels = data_train
      self.val_x, self.val_labels = data_val
    elif data:
      data_train,data_val=my_train_val_split(data,val_split,seed)
      self.train_x, self.train_labels = data_train
      self.val_x, self.val_labels = data_val
    else:
      raise


  def train(self, epoch, batch_size, lr, metrics = "Acc"):
    for ep in range(epoch):
      lst_ind_train=len(self.train_x)-len(self.train_x)%batch_size if self.drop_last else len(self.train_x)
      for i in range(0,lst_ind_train,batch_size):
        xb = self.train_x[i:i+batch_size]
        yb = self.train_labels[i:i+batch_size]

        # forward pass
        out = self.net.forward(xb)
        loss = self.criterion.forward(out,yb)
        self.optim.step(self.net, self.criterion, loss, lr)
      if metrics == "Acc":
        print(f"Epoch {ep}: train_acc = {accuracy(self.net,self.train_x,self.train_labels)} val_acc = {accuracy(self.net,self.val_x,self.val_labels)}")
      elif metrics == "Loss":
        print(f"Epoch {ep}: train_Loss = {Loss(self.net,self.criterion,self.train_x,self.train_labels)} val_Loss = {Loss(self.net,self.criterion,self.val_x,self.val_labels)}")

def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = plt.subplots(1, 1)
    #pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')

    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha = 0.5)
    plt.show()