"""
@Title:
    hp-VPINNs: A General Framework For Solving PDEs
    Application to 2D Poisson Eqn in L-Shaped Domain

@author: 
    Ehsan Kharazmi
    Division of Applied Mathematics
    Brown University
    ehsan_kharazmi@brown.edu

"""
###############################################################################
###############################################################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time

np.random.seed(1234)
tf.set_random_seed(1234)
###############################################################################
###############################################################################
class VPINN:
    def __init__(self, X_u_train, u_train, X_f_train, f_train, X_quad, W_quad,\
                 F_exact_total, elements, N_testfcn_total, X_test, u_test, layers):

        self.x = X_u_train[:,0:1]
        self.y = X_u_train[:,1:2]
        self.utrain = u_train
        self.xquad  = X_quad[:,0:1]
        self.yquad  = X_quad[:,1:2]
        self.wquad  = W_quad
        self.xf = X_f_train[:,0:1]
        self.yf = X_f_train[:,1:2]
        self.ftrain = f_train
        self.xtest = X_test[:,0:1]
        self.ytest = X_test[:,1:2]
        self.utest = u_test
        self.NE_total = np.shape(N_testfcn_total)[0]
        self.Ntestx = N_testfcn_total[0][0]
        self.Ntesty = N_testfcn_total[0][1]
        self.F_ext_total = F_exact_total
       
        self.layers = layers
        self.weights, self.biases, self.a = self.initialize_NN(layers)
        
        self.x_tf     = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.y_tf     = tf.placeholder(tf.float64, shape=[None, self.y.shape[1]])
        self.u_tf     = tf.placeholder(tf.float64, shape=[None, self.utrain.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.xf.shape[1]])
        self.y_f_tf = tf.placeholder(tf.float64, shape=[None, self.yf.shape[1]])
        self.f_tf   = tf.placeholder(tf.float64, shape=[None, self.ftrain.shape[1]])
        self.x_test = tf.placeholder(tf.float64, shape=[None, self.xtest.shape[1]])
        self.y_test = tf.placeholder(tf.float64, shape=[None, self.ytest.shape[1]])
        self.x_quad = tf.placeholder(tf.float64, shape=[None, self.xquad.shape[1]])
        self.y_quad = tf.placeholder(tf.float64, shape=[None, self.yquad.shape[1]])
                 

        self.u_pred_boundary = self.net_u(self.x_tf, self.y_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.y_f_tf)
        self.u_test = self.net_u(self.x_test, self.y_test)
        
        self.varloss_total = 0
        self.U_NN_total = []
        for e in range(self.NE_total):
            F_ext_element  = self.F_ext_total[e]
            Ntest_elementx = N_testfcn_total[e][0]
            Ntest_elementy = N_testfcn_total[e][1]
            
            grid_x_element = np.array([elements[e,0,0], elements[e,1,0]])
            grid_y_element = np.array([elements[e,0,1], elements[e,2,1]])
            
            x_quad_element = tf.constant(grid_x_element[0] + (grid_x_element[1] - grid_x_element[0])/2*(self.xquad+1))
            y_quad_element = tf.constant(grid_y_element[0] + (grid_y_element[1] - grid_y_element[0])/2*(self.yquad+1))
            jacobian_x     = ((grid_x_element[1]-grid_x_element[0])/2)
            jacobian_y     = ((grid_y_element[1]-grid_y_element[0])/2)
            jacobian       = ((grid_x_element[1] - grid_x_element[0])/2) * ((grid_y_element[1] - grid_y_element[0])/2)
            
            d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, y_quad_element)
            d1yu_NN_quad_element, d2yu_NN_quad_element = self.net_dyu(x_quad_element, y_quad_element)
                            
            testx_quad_element = self.Test_fcnx(Ntest_elementx, self.xquad)
            d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
            testy_quad_element = self.Test_fcny(Ntest_elementy, self.yquad)
            d1testy_quad_element, d2testy_quad_element = self.dTest_fcn(Ntest_elementy, self.yquad)
    

            integrand_1 = d2xu_NN_quad_element + d2yu_NN_quad_element
            
            if var_form == 0:
                U_NN_element = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*integrand_1) \
                                for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)

            if var_form == 1:
                U_NN_element_1 = tf.convert_to_tensor([[jacobian/jacobian_x*tf.reduce_sum(\
                                self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*d1xu_NN_quad_element) \
                                for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                U_NN_element_2 = tf.convert_to_tensor([[jacobian/jacobian_y*tf.reduce_sum(\
                                self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d1testy_quad_element[k]*d1yu_NN_quad_element) \
                                for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                U_NN_element = - U_NN_element_1 - U_NN_element_2



    
            self.U_NN_total.append(U_NN_element)
            Res_NN_element = tf.reshape(U_NN_element - F_ext_element, [1,-1]) 
            loss_element = tf.reduce_mean(tf.square(Res_NN_element))
            self.varloss_total = self.varloss_total + loss_element


        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_pred_boundary))
        self.lossv = self.varloss_total
       
        self.loss  = 10*self.lossb + self.lossv 

        
        self.optimizer_Adam = tf.train.AdamOptimizer(0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
###############################################################################
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            a = tf.Variable(0.01, dtype=tf.float64)
            weights.append(W)
            biases.append(b)        
        return weights, biases, a
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim), dtype=np.float64)
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)
 
    
    def neural_net(self, X, weights, biases, a):
        num_layers = len(weights) + 1        
        H = X 
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, y):  
        u = self.neural_net(tf.concat([x,y],1), self.weights, self.biases,  self.a)
        return u

    def net_dxu(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        return d1xu, d2xu

    def net_dyu(self, x, y):
        u   = self.net_u(x, y)
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        return d1yu, d2yu

    def net_f(self, x, y):
        u   = self.net_u(x, y)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        d1yu = tf.gradients(u, y)[0]
        d2yu = tf.gradients(d1yu, y)[0]
        ftemp = d2xu + d2yu
        return ftemp

    def Test_fcnx(self, N_test,x):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
            test_total.append(test)
        return np.asarray(test_total)

    def Test_fcny(self, N_test,y):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
            test_total.append(test)
        return np.asarray(test_total)


    def dTest_fcn(self, N_test,x):
        d1test_total = []
        d2test_total = []
        for n in range(1,N_test+1):
            if n==1:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)
            elif n==2:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
            else:
                d1test = ((n+2)/2)*Jacobi(n,1,1,x) - ((n)/2)*Jacobi(n-2,1,1,x)
                d2test = ((n+2)*(n+3)/(2*2))*Jacobi(n-1,2,2,x) - ((n)*(n+1)/(2*2))*Jacobi(n-3,2,2,x)
                d1test_total.append(d1test)
                d2test_total.append(d2test)    
        return np.asarray(d1test_total), np.asarray(d2test_total)
        
###############################################################################
    def train(self, nIter):
        
        tf_dict = {self.x_tf: self.x  , self.y_tf: self.y,
                   self.u_tf: self.utrain,
                   self.x_test: self.xtest, self.y_test: self.ytest,
                   self.x_f_tf: self.xf, self.y_f_tf: self.yf}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            loss_his.append(loss_value)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Loss: %.3e, Time: %.2f'])
                print(str_print % (it, loss_value, elapsed))
                start_time = time.time()

    def predict(self):
        u_pred = self.sess.run(self.u_test, {self.x_test: self.xtest, self.y_test: self.ytest})
        return u_pred

#%%
##############################################################################
# =============================================================================
#                               Main
# =============================================================================    
if __name__ == "__main__":     

    '''
    Hyper-parameters: 
        scheme     = is 'hp-VPINNs'
        Net_layer  = the structure of fully connected network
        var_form   = the form of the variational formulation used in VPINNs
                     0, 1, 2: no, once, twice integration-by-parts
        NE_total   = the total number of elements in the domain
        N_testfcn_x, N_testfcn_y = number of test functions in x and y direction
        N_quad     = number of quadrature points in each direction in each element
        N_bound    = number of boundary points in the boundary loss
        N_residual = number of residual points in PINNs
        delta      = a parameter for the case of NE_total = 8
    '''
    scheme = 'VPINNs'
    Net_layer = [2] + [5] * 3 + [1]
    var_form  = 0
    NE_total = 3
    N_testfcn_x = 5
    N_testfcn_y = 5
    N_quad = 10
    N_bound = 20
    N_residual = 100    
    delta = 0.05

    ###########################################################################
    def Test_fcn_x(n,x):
       test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
       return test
    def Test_fcn_y(n,y):
       test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
       return test
    ###########################################################################    
    def f_ext(x,y):
        return -1
    ###########################################################################
    # Boundary points
    x_up = lhs(1,N_bound)-1
    y_up = np.empty(len(x_up))[:,None]
    y_up.fill(1)
    b_up = np.empty(len(x_up))[:,None]
    b_up.fill(0.0)
    x_up_train = np.hstack((x_up, y_up))
    u_up_train = b_up
    
    x_up2 = lhs(1,N_bound)
    y_up2 = np.empty(len(x_up))[:,None]
    y_up2.fill(0.0)
    b_up2 = np.empty(len(x_up2))[:,None]
    b_up2.fill(0.0)
    x_up2_train = np.hstack((x_up2, y_up2))
    u_up2_train = b_up2

    x_lo = 2*lhs(1,2*N_bound)-1
    y_lo = np.empty(len(x_lo))[:,None]
    y_lo.fill(-1)
    b_lo = np.empty(len(x_lo))[:,None]
    b_lo.fill(0.0)
    x_lo_train = np.hstack((x_lo, y_lo))
    u_lo_train = b_lo

    y_ri = lhs(1,N_bound)-1
    x_ri = np.empty(len(y_ri))[:,None]
    x_ri.fill(1)
    b_ri = np.empty(len(y_ri))[:,None]
    b_ri.fill(0.0)
    x_ri_train = np.hstack((x_ri, y_ri))
    u_ri_train = b_ri    


    y_ri2 = lhs(1,N_bound)
    x_ri2 = np.empty(len(y_ri2))[:,None]
    x_ri2.fill(0)
    b_ri2 = np.empty(len(y_ri2))[:,None]
    b_ri2.fill(0.0)
    x_ri2_train = np.hstack((x_ri2, y_ri2))
    u_ri2_train = b_ri2

    y_le = 2*lhs(1,2*N_bound)-1
    x_le = np.empty(len(y_le))[:,None]
    x_le.fill(-1)
    b_le = np.empty(len(y_le))[:,None]
    b_le.fill(0.0)
    x_le_train = np.hstack((x_le, y_le))
    u_le_train = b_le    

    X_u_train = np.concatenate((x_up_train, x_up2_train, x_lo_train, x_ri_train, x_ri2_train, x_le_train))
    u_train = np.concatenate((u_up_train, u_up2_train, u_lo_train, u_ri_train, u_ri2_train, u_le_train))

    ###########################################################################
    # Residual points for PINNs
    grid_pt = lhs(2,N_residual)
    xf = 2*grid_pt[:,0]-1
    yf = 2*grid_pt[:,1]-1
    ff = np.asarray([ f_ext(xf[j],yf[j]) for j in range(len(yf))])
    X_f_train = np.hstack((xf[:,None],yf[:,None]))
    f_train = ff[:,None]

    ###########################################################################
    # Quadrature points
    [X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    Y_quad, WY_quad   = (X_quad, WX_quad)
    xx, yy            = np.meshgrid(X_quad,  Y_quad)
    wxx, wyy          = np.meshgrid(WX_quad, WY_quad)
    XY_quad_train     = np.hstack((xx.flatten()[:,None],  yy.flatten()[:,None]))
    WXY_quad_train    = np.hstack((wxx.flatten()[:,None], wyy.flatten()[:,None]))

    ###########################################################################
    # Domain Decomposition
    if NE_total == 3:
        vertices = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1]] #[x, y]
        elements = np.asarray(
                    [[vertices[0],vertices[1],vertices[3],vertices[4]]
                    ,[vertices[1],vertices[2],vertices[4],vertices[5]]
                    ,[vertices[3],vertices[4],vertices[6],vertices[7]]])
        N_testfcn_total = [[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y]]

    if NE_total == 8:
        vertices = [[-1,-1],[-delta,-1],[1,-1],\
                    [-1,-delta],[-delta,-delta],[0,-delta],[delta,-delta],[1,-delta],\
                    [-delta,0],[0,0],[delta,0],[1,0],\
                    [-delta,delta],[0,delta],\
                    [-1,1],[-delta,1],[0,1]]
        elements = np.asarray(
                    [[vertices[0],vertices[1],vertices[3],vertices[4]]
                    ,[vertices[1],vertices[2],vertices[4],vertices[7]]
                    ,[vertices[3],vertices[4],vertices[14],vertices[15]]
                    ,[vertices[4],vertices[5],vertices[8],vertices[9]]
                    ,[vertices[5],vertices[6],vertices[9],vertices[10]]
                    ,[vertices[6],vertices[7],vertices[10],vertices[11]]
                    ,[vertices[8],vertices[9],vertices[12],vertices[13]]
                    ,[vertices[12],vertices[13],vertices[15],vertices[16]]
                    ])
        N_testfcn_total = [[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]]
        
    if NE_total == 6:
        vertices = [[-1,-1],[-0.5,-1],[0.0,-1],[1,-1],\
                    [-1,-0.5],[-0.5,-0.5],[0,-0.5],\
                    [-1,0.0],[-0.5,0.0],[0,0.0],[1,0.0],\
                    [-1,1],[0,1]]
        elements = np.asarray(
                    [[vertices[0],vertices[1],vertices[4],vertices[5]]
                    ,[vertices[1],vertices[2],vertices[5],vertices[6]]
                    ,[vertices[4],vertices[5],vertices[7],vertices[8]]
                    ,[vertices[5],vertices[6],vertices[8],vertices[9]]
                    ,[vertices[2],vertices[3],vertices[9],vertices[10]]
                    ,[vertices[7],vertices[9],vertices[11],vertices[12]]
                    ])
        N_testfcn_total = [[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]]

    if NE_total == 18:
        vertices = [[-1,-1]   ,[-0.75,-1]   ,[-0.5,-1]   ,[-0.25,-1]   ,[0.0,-1]   ,[1,-1] ,\
                    [-1,-0.75],[-0.75,-0.75],[-0.5,-0.75],[-0.25,-0.75],[0.0,-0.75],        \
                    [-1,-0.5] ,[-0.75,-0.5] ,[-0.5,-0.5] ,[-0.25,-0.5] ,[0.0,-0.5] ,        \
                    [-1,-0.25],[-0.75,-0.25],[-0.5,-0.25],[-0.25,-0.25],[0.0,-0.25],        \
                    [-1,0.0]  ,[-0.75,0.0]  ,[-0.5,0.0]  ,[-0.25,0.0]  ,[0.0,0.0]  ,[1,0.0],\
                    [-1,1]    ,[0,1]]
        elements = np.asarray(
                    [[vertices[0],vertices[1],vertices[6],vertices[7]]
                    ,[vertices[1],vertices[2],vertices[7],vertices[8]]
                    ,[vertices[2],vertices[3],vertices[8],vertices[9]]
                    ,[vertices[3],vertices[4],vertices[9],vertices[10]]
                    ,[vertices[6],vertices[7],vertices[11],vertices[12]]
                    ,[vertices[7],vertices[8],vertices[12],vertices[13]]
                    ,[vertices[8],vertices[9],vertices[13],vertices[14]]
                    ,[vertices[9],vertices[10],vertices[14],vertices[15]]
                    ,[vertices[11],vertices[12],vertices[16],vertices[17]]
                    ,[vertices[12],vertices[13],vertices[17],vertices[18]]
                    ,[vertices[13],vertices[14],vertices[18],vertices[19]]
                    ,[vertices[14],vertices[15],vertices[19],vertices[20]]
                    ,[vertices[16],vertices[17],vertices[21],vertices[22]]
                    ,[vertices[17],vertices[18],vertices[22],vertices[23]]
                    ,[vertices[18],vertices[19],vertices[23],vertices[24]]
                    ,[vertices[19],vertices[20],vertices[24],vertices[25]]
                    ,[vertices[4],vertices[5],vertices[25],vertices[26]]
                    ,[vertices[21],vertices[25],vertices[27],vertices[28]]
                    ])
        N_testfcn_total = [[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]]

    if NE_total == 35:
        vertices = [[-1,-1]   ,[-0.75,-1]   ,[-0.5,-1]   ,[-0.25,-1]   ,[0.0,-1]   ,[0.25,-1]   ,[1,-1] ,\
                    [-1,-0.75],[-0.75,-0.75],[-0.5,-0.75],[-0.25,-0.75],[0.0,-0.75],[0.25,-0.75],        \
                    [-1,-0.5] ,[-0.75,-0.5] ,[-0.5,-0.5] ,[-0.25,-0.5] ,[0.0,-0.5] ,[0.25,-0.5] ,        \
                    
                    [-1,-0.25],[-0.75,-0.25],[-0.5,-0.25],[-0.25,-0.25] ,[-0.125,-0.25] ,[0.0,-0.25] ,[0.125,-0.25] ,[0.25,-0.25] ,           \
                                                          [-0.25,-0.125],[-0.125,-0.125],[0.0,-0.125],[0.125,-0.125],[0.25,-0.125],           \
                    [-1,0.0]  ,[-0.75,0.0]  ,[-0.5,0.0]  ,[-0.25,0.0]   ,[-0.125,0.0]   ,[0.0,0.0]   ,[0.125,0.0]   ,[0.25,0.0]   ,[1.0,0.0], \
                                                          [-0.25,0.125] ,[-0.125,0.125] ,[0.0,0.125],\
                    [-1,0.25] ,[-0.75,0.25] ,[-0.5,0.25] ,[-0.25,0.25]  ,[-0.125,0.25]  ,[0.0,0.25] ,\
                    [-1,1]    ,[0,1]]
        elements = np.asarray(
                    [[vertices[0],vertices[1],vertices[7] ,vertices[8]]
                    ,[vertices[1],vertices[2],vertices[8] ,vertices[9]]
                    ,[vertices[2],vertices[3],vertices[9] ,vertices[10]]
                    ,[vertices[3],vertices[4],vertices[10],vertices[11]]
                    ,[vertices[4],vertices[5],vertices[11],vertices[12]]

                    ,[vertices[7],vertices[8] ,vertices[13] ,vertices[14]]
                    ,[vertices[8],vertices[9] ,vertices[14] ,vertices[15]]
                    ,[vertices[9],vertices[10],vertices[15] ,vertices[16]]
                    ,[vertices[10],vertices[11],vertices[16],vertices[17]]
                    ,[vertices[11],vertices[12],vertices[17],vertices[18]]

                    ,[vertices[13],vertices[14],vertices[19],vertices[20]]
                    ,[vertices[14],vertices[15],vertices[20],vertices[21]]
                    ,[vertices[15],vertices[16],vertices[21],vertices[22]]
                    ,[vertices[16],vertices[17],vertices[22],vertices[24]]
                    ,[vertices[17],vertices[18],vertices[24],vertices[26]]

                    ,[vertices[19],vertices[20],vertices[32],vertices[33]]
                    ,[vertices[20],vertices[21],vertices[33],vertices[34]]
                    ,[vertices[21],vertices[22],vertices[34],vertices[35]]

                    ,[vertices[32],vertices[33],vertices[44],vertices[45]]
                    ,[vertices[33],vertices[34],vertices[45],vertices[46]]
                    ,[vertices[34],vertices[35],vertices[46],vertices[47]]

                    ,[vertices[22],vertices[23],vertices[27],vertices[28]]
                    ,[vertices[23],vertices[24],vertices[28],vertices[29]]
                    ,[vertices[24],vertices[25],vertices[29],vertices[30]]
                    ,[vertices[25],vertices[26],vertices[30],vertices[31]]

                    ,[vertices[27],vertices[28],vertices[35],vertices[36]]
                    ,[vertices[28],vertices[29],vertices[36],vertices[37]]
                    ,[vertices[29],vertices[30],vertices[37],vertices[38]]
                    ,[vertices[30],vertices[31],vertices[38],vertices[39]]

                    ,[vertices[35],vertices[36],vertices[41],vertices[42]]
                    ,[vertices[36],vertices[37],vertices[42],vertices[43]]

                    ,[vertices[41],vertices[42],vertices[47],vertices[48]]
                    ,[vertices[42],vertices[43],vertices[48],vertices[49]]

                    ,[vertices[5],vertices[6],vertices[39],vertices[40]]
                    ,[vertices[44],vertices[49],vertices[50],vertices[51]]
                    ])

        N_testfcn_total = [[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y],[N_testfcn_x, N_testfcn_y]
                          ,[N_testfcn_x, N_testfcn_y]]

    # Construction of RHS for VPINNs    
    #+++++++++++++++++++
    x_quad  = XY_quad_train[:,0:1]
    y_quad  = XY_quad_train[:,1:2]
    w_quad  = WXY_quad_train
    F_ext_total = []
    
    for e in range(NE_total):
        Ntest_elementx  = N_testfcn_total[e][0]
        Ntest_elementy  = N_testfcn_total[e][1]
        grid_x_element = np.array([elements[e,0,0], elements[e,1,0]])
        grid_y_element = np.array([elements[e,0,1], elements[e,2,1]])
        x_quad_element = grid_x_element[0] + (grid_x_element[1] - grid_x_element[0])/2*(x_quad+1)
        y_quad_element = grid_y_element[0] + (grid_y_element[1] - grid_y_element[0])/2*(y_quad+1)
        jacobian       = ((grid_x_element[1] - grid_x_element[0])/2) * ((grid_y_element[1] - grid_y_element[0])/2)        
        testx_quad_element = np.asarray([ Test_fcn_x(n,x_quad)  for n in range(1, Ntest_elementx+1)])
        testy_quad_element = np.asarray([ Test_fcn_y(n,y_quad)  for n in range(1, Ntest_elementy+1)])
        f_quad_element = f_ext(x_quad_element, y_quad_element)
        F_ext_element = np.asarray([[jacobian*np.sum(\
                        w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element) \
                        for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
        F_ext_total.append(F_ext_element)

    ###########################################################################
    # Test points
    X_test_data = np.load('Data/X_test.npy')
    u_test_data = np.load('Data/y_ref.npy')    
    X_test_total = X_test_data
    u_test = u_test_data
    ytest = 1

    ###########################################################################
    model = VPINN(X_u_train, u_train, X_f_train, f_train, XY_quad_train, WXY_quad_train,\
                  F_ext_total, elements, N_testfcn_total, X_test_total, u_test, Net_layer)
    
    u_pred_his, loss_his = [], []
    model.train(25000 + 1)
    u_pred = model.predict()


#%%
    ###########################################################################
    # =============================================================================
    #    Plotting
    # =============================================================================
  
    fontsize = 24
    fig = plt.figure(1)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(loss_his)
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=9,h=9)
    plt.savefig(''.join(['Poisson2D_',scheme,'_loss','.pdf']))

    ###########################################################################
    fontsize = 32
    fig, ax = plt.subplots(constrained_layout=True)
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel('$x$', fontsize = fontsize)
    plt.ylabel('$y$', fontsize = fontsize)
    
    for e in range(len(elements)):
        ax.add_patch(Polygon(np.concatenate([elements[e,0:2], elements[e,-1:1:-1]]), closed=True,
                          fill=False, edgecolor='k'))
    ax.locator_params(nbins=5)
    plt.tick_params( labelsize = 26)
    #fig.tight_layout()
    fig.set_size_inches(w=9,h=9)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Domain','.pdf']))



    ###########################################################################    
    x_train_plot, y_train_plot = zip(*X_u_train)
    x_f_plot, y_f_plot = zip(*X_f_train)
    
    x_test_plot = np.asarray(np.split(X_test_total[:,0:1].flatten(),161))
    y_test_plot = np.asarray(np.split(X_test_total[:,1:2].flatten(),161))
    u_test_plot = np.asarray(np.split(u_test.flatten(),161))
    
    u_pred_plot = np.asarray(np.split(u_pred.flatten(),161))
    for i in range(81,161,1):
        for j in range(81,161,1):
            u_pred_plot[i][j] = u_test_plot[i][j]
    
    
    labelsize = 26    
    fig_ext, ax_ext = plt.subplots(constrained_layout=True)
    CS_ext = ax_ext.contourf(x_test_plot, y_test_plot, u_test_plot, 100, cmap='jet', origin='lower')
    cbar = fig_ext.colorbar(CS_ext, shrink=0.73, ticks=[0,0.04, 0.08, 0.12, 0.16])
    cbar.ax.tick_params(labelsize = labelsize)
    ax_ext.locator_params(nbins=5)
    ax_ext.set_xlabel('$x$' , fontsize = fontsize)
    ax_ext.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_ext.set_aspect(1)
    plt.box(False)
    fig_ext.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Exact','.png']))
        
    fig_pred, ax_pred = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred.contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred.colorbar(CS_pred, shrink=0.71, ticks=[0,0.04, 0.08, 0.12, 0.16])
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred.locator_params(nbins=5)
    ax_pred.set_xlabel('$x$' , fontsize = fontsize)
    ax_pred.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_pred.set_aspect(1)
    for e in range(len(elements)):
        ax_pred.add_patch(Polygon(np.concatenate([elements[e,0:2], elements[e,-1:1:-1]]), closed=True,
                          fill=False, edgecolor='w'))
    plt.box(False)
    fig_pred.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Predict','.png']))
    
    
    fig_err, ax_err = plt.subplots(constrained_layout=True)
    levels = np.linspace(0, 0.025, 100)
    CS_err = ax_err.contourf(x_test_plot, y_test_plot, abs(u_test_plot - u_pred_plot), levels = levels, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err, shrink=0.73, ticks=[0, 0.01, 0.02, 0.03, 0.04])
    cbar.ax.tick_params(labelsize = labelsize)
    ax_err.set_xlabel('$x$' , fontsize = fontsize)
    ax_err.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_err.set_aspect(1)
    ax_err.locator_params(nbins=5)
    for e in range(len(elements)):
        ax_err.add_patch(Polygon(np.concatenate([elements[e,0:2], elements[e,-1:1:-1]]), closed=True,
                          fill=False, edgecolor='w'))
    plt.box(False)
    fig_err.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_PntErr','.png']))
    

    fig_err, ax_err = plt.subplots(constrained_layout=True)
    levels = np.linspace(0, 0.04, 100)
    CS_err = ax_err.contourf(x_test_plot, y_test_plot, abs(u_test_plot - u_pred_plot), levels = levels, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err, shrink=0.73, ticks=[0, 0.01, 0.02, 0.03, 0.04])
    cbar.ax.tick_params(labelsize = labelsize)
    ax_err.set_xlim(-0.42,0.42)
    ax_err.set_ylim(-0.42,0.42)
    ax_err.set_xlabel('$x$' , fontsize = fontsize)
    ax_err.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_err.set_aspect(1)
    ax_err.locator_params(nbins=5)
    for e in range(len(elements)):
        ax_err.add_patch(Polygon(np.concatenate([elements[e,0:2], elements[e,-1:1:-1]]), closed=True,
                          fill=False, edgecolor='w'))
    ax_err.add_patch(Polygon(np.array([[0,0],[0.01,0],[0.01,0.01],[0,0.01]]), closed=True,
                          fill=True, color = 'w', edgecolor='w'))
    plt.box(False)
    fig_err.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_PntErr_zoom1','.png']))
    
    
    fig_err, ax_err = plt.subplots(constrained_layout=True)
    levels = np.linspace(0, 0.04, 100)
    CS_err = ax_err.contourf(x_test_plot, y_test_plot, abs(u_test_plot - u_pred_plot), levels = levels, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err, shrink=0.73, ticks=[0, 0.01, 0.02, 0.03, 0.04])
    cbar.ax.tick_params(labelsize = labelsize)
    ax_err.set_xlim(-0.035,0.035)
    ax_err.set_ylim(-0.035,0.035)
    ax_err.set_xlabel('$x$' , fontsize = fontsize)
    ax_err.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_err.set_aspect(1)
    ax_err.locator_params(nbins=5)
    for e in range(len(elements)):
        ax_err.add_patch(Polygon(np.concatenate([elements[e,0:2], elements[e,-1:1:-1]]), closed=True,
                          fill=False, edgecolor='w'))
    ax_err.add_patch(Polygon(np.array([[0,0],[0.01,0],[0.01,0.01],[0,0.01]]), closed=True,
                          fill=True, color = 'w', edgecolor='w'))
    plt.box(False)
    fig_err.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_PntErr_zoom2','.png']))

