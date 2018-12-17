import os, sys, struct, warnings, random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict    # used for data structure
from mnist import load_mnist           # load mnist.py under the same directary
sys.path.append(os.pardir)



class Sigmoid:
    def __init(self):
        self.loss= None
        self.y   = None
        self.t   = None
    def forward(self, x, t):
        self.y   = sigmoid(x)
        self.t   = t
        self.loss= cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx         = (self.y - self.t) / batch_size
        return dx


class MatrixMulti:
    def __init__(self, W, b):
        self.W  = W
        self.b  = b
        self.x  = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x  = x
        out     = np.dot(x, self.W)+ self.b
        return out

    def backward(self, dout):
        dx      = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx


class Relu:
    def __init(self):
        self.mask      = None

matplotlib==3.0.0
numpy==1.15.2
    def forward(self, x):
        self.mask      = (x <= 0)
        out            = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask]= 0
        dx             = dout
        return dx



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1']           = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1']           = np.zeros(hidden_size)
        self.params['W2']           = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2']           = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['MatrixMulti1'] = MatrixMulti(self.params['W1'], self.params['b1'])
        self.layers['Relu1']        = Relu()
        self.layers['MatrixMulti2'] = MatrixMulti(self.params['W2'], self.params['b2'])
        
        #self.y                      = None
        
        self.lastLayer              = Sigmoid()


    def predict(self, x, layer=-1):
        i = 0
        for layers in self.layers.values():
            if i == layer:
                break
            x = layers.forward(x)
            i += 1

        return x


    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)  
    

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout     = 1
        dout     = self.lastLayer.backward(dout)

        layers   = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads    = {}
        grads['W1'], grads['b1'] = self.layers['MatrixMulti1'].dW, self.layers['MatrixMulti1'].db
        grads['W2'], grads['b2'] = self.layers['MatrixMulti2'].dW, self.layers['MatrixMulti2'].db

        return grads




def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t      = t.reshape(1, t.size)
        y      = y.reshape(1, y.size)

    h = 0.000001 
    return -np.sum(t * np.log(y + h)+ (1 - t)* np.log(1 - y + h))





(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


network         = NeuralNetwork(input_size=784, hidden_size=128, output_size=784)

iters_num       = 10000
train_size      = x_train.shape[0]
test_size       = x_test.shape[0]
batch_size      = 64
learning_rate   = 0.01

train_loss_list = []
test_loss_list  = []


for i in range(iters_num):
    train_batch       = np.random.choice(train_size, batch_size)
    test_batch        = np.random.choice(test_size, batch_size)
    x_train_batch     = x_train[train_batch]
    t_train_batch     = t_train[train_batch]
    x_test_batch      = x_test[test_batch]
    t_test_batch      = t_test[test_batch]  
  
    grad              = network.gradient(x_train_batch, x_train_batch)
     
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss_train = network.loss(x_train_batch, x_train_batch)
    loss_test  = network.loss(x_test_batch, x_test_batch)
    train_loss_list.append(loss_train)
    test_loss_list.append(loss_test)


#   -----------------NN structure; loss curve ----------------------------

loss_train     = np.array(train_loss_list)
#print('loss_train:', loss_train)
plt.plot(np.arange(iters_num), loss_train, linewidth = 0.5, markersize = 0.5)
plt.title('loss curve: train')
plt.savefig('num2_loss_curve_train.png')
plt.show()

loss_test      = np.array(test_loss_list)
#print("loss_test", loss_test)
plt.plot(np.arange(iters_num), loss_test, linewidth = 0.5, markersize = 0.5)
plt.title('loss curve: test')
plt.savefig('num2_loss_curve_test.png')
plt.show()

#   ------------------Dimension reduction-----------------------

result = network.predict(x_train)
hidden = network.predict(x_train, layer=2)
#print(result.shape) 60000,784
#print(hidden.shape) 60000,128

plt.figure()


def output(x_train, hidden_data):
    hidden = hidden_data[:, :2]

    arr_x= np.array(hidden[:,0])
    arr_y= np.array(hidden[:,1])
    x_max= max(arr_x)
    x_min= min(arr_x)
    y_max= max(arr_y)
    y_min= min(arr_x)

    x_r= (x_max- x_min)/ 5
    x01= np.logical_and(arr_x>= x_min, arr_x< x_min+ x_r)
    x12= np.logical_and(arr_x>= x_min+ x_r, arr_x< x_min+ 2* x_r)
    x23= np.logical_and(arr_x>= x_min+ 2* x_r, arr_x< x_min+ 3* x_r)
    x34= np.logical_and(arr_x>= x_min+ 3* x_r, arr_x< x_min+ 4* x_r)
    x45= np.logical_and(arr_x>= x_min+ 4* x_r, arr_x<= x_max)
    x_ar= (x01, x12, x23, x34, x45)
    x_ar= np.array(x_ar)

    y_r= (y_max- y_min)/ 5
    y01= np.logical_and(arr_y>= y_min, arr_y< y_min+ y_r)
    y12= np.logical_and(arr_y>= y_min+ y_r, arr_y< y_min+ 2* y_r)
    y23= np.logical_and(arr_y>= y_min+ 2* y_r, arr_y< y_min+ 3* y_r)
    y34= np.logical_and(arr_y>= y_min+ 3* y_r, arr_y< y_min+ 4* y_r)
    y45= np.logical_and(arr_y>= y_min+ 4* y_r, arr_y<= y_max)
    y_ar= (y01, y12, y23, y34, y45)
    y_ar= np.array(y_ar)


    fig, ax= plt.subplots()

    for i in range(5):
        for j in range(5):
            re= np.logical_and(x_ar[i], y_ar[j])
            #print(re.shape)
            if True in re:
                ax.imshow(x_train[re][0,:].reshape(28, 28), cmap='gray', extent=(hidden[re][0, 0], hidden[re][0, 0]+ x_r* 0.5, hidden[re][0, 1], hidden[re][0, 1]+ y_r* 0.5), zorder= 30)
                ax.scatter(hidden[re][0, 0], hidden[re][0, 1], c='r', s=5)

    plt.scatter(arr_x, arr_y, c='g', zorder= -1, s=5)
    plt.savefig('dimension_reduction.png')
    plt.show()


output(x_train, hidden)

#---------------------Reconstruction results and filters----------------
plt.subplot(2,2,1)
plt.imshow(result[0].reshape(28,-1), cmap='gray')
plt.subplot(2,2,2)
plt.imshow(result[1].reshape(28,-1), cmap='gray')
plt.subplot(2,2,3)
plt.imshow(x_train[0].reshape(28,-1), cmap='gray')
plt.subplot(2,2,4)
plt.imshow(x_train[1].reshape(28,-1), cmap='gray')
plt.savefig('reconstruction_result1.png')
plt.show()


plt.subplot(2,2,1)
plt.imshow(result[10000].reshape(28,-1), cmap='gray')
plt.subplot(2,2,2)
plt.imshow(result[10001].reshape(28,-1), cmap='gray')
plt.subplot(2,2,3)
plt.imshow(x_train[10000].reshape(28,-1), cmap='gray')
plt.subplot(2,2,4)
plt.imshow(x_train[10001].reshape(28,-1), cmap='gray')
plt.savefig('reconstruction_result2.png')
plt.show()


plt.subplot(2,2,1)
plt.imshow(result[40000].reshape(28,-1), cmap='gray')
plt.subplot(2,2,2)
plt.imshow(result[40001].reshape(28,-1), cmap='gray')
plt.subplot(2,2,3)
plt.imshow(x_train[40000].reshape(28,-1), cmap='gray')
plt.subplot(2,2,4)
plt.imshow(x_train[40001].reshape(28,-1), cmap='gray')
plt.savefig('reconstruction_result3.png')
plt.show()


#print(network.params['W1'].shape) 784,128
filters = network.params['W1'][:,:16]
#print(filters.shape)
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(filters[:,i].reshape(28,-1), cmap='gray')
plt.savefig('filters.png')
plt.show()

