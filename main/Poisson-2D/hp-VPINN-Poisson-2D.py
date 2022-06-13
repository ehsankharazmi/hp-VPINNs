"""
@Title:
    hp-VPINNs: A General Framework For Solving PDEs
    Application to 2D Poisson Eqn

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
from pyDOE import lhs
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time

np.random.seed(1234)
tf.set_random_seed(1234)
###############################################################################
###############################################################################
class VPINN:
    def __init__(self, X_u_train, u_train, X_f_train, f_train, X_quad, W_quad, U_exact_total, F_exact_total,\
                 gridx, gridy, N_testfcn, X_test, u_test, layers):

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
        self.Nelementx = np.size(N_testfcn[0])
        self.Nelementy = np.size(N_testfcn[1])
        self.Ntestx = N_testfcn[0][0]
        self.Ntesty = N_testfcn[1][0]
        self.U_ext_total = U_exact_total
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
        for ex in range(self.Nelementx):
            for ey in range(self.Nelementy):
                F_ext_element  = self.F_ext_total[ex, ey]
                Ntest_elementx = N_testfcn[0][ex]
                Ntest_elementy = N_testfcn[1][ey]
                
                x_quad_element = tf.constant(gridx[ex] + (gridx[ex+1]-gridx[ex])/2*(self.xquad+1))
                y_quad_element = tf.constant(gridy[ey] + (gridy[ey+1]-gridy[ey])/2*(self.yquad+1))
                jacobian_x     = ((gridx[ex+1]-gridx[ex])/2)
                jacobian_y     = ((gridy[ey+1]-gridy[ey])/2)
                jacobian       = ((gridx[ex+1]-gridx[ex])/2)*((gridy[ey+1]-gridy[ey])/2)
                
                u_NN_quad_element = self.net_u(x_quad_element, y_quad_element)
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

    
                if var_form == 2:
                    U_NN_element_1 = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*d2testx_quad_element[r]*self.wquad[:,1:2]*testy_quad_element[k]*u_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    U_NN_element_2 = tf.convert_to_tensor([[jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*d2testy_quad_element[k]*u_NN_quad_element) \
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementy)], dtype= tf.float64)
                    U_NN_element = U_NN_element_1 + U_NN_element_2


                Res_NN_element = tf.reshape(U_NN_element - F_ext_element, [1,-1]) 
                loss_element = tf.reduce_mean(tf.square(Res_NN_element))
                self.varloss_total = self.varloss_total + loss_element
 
        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_pred_boundary))
        self.lossv = self.varloss_total
        self.lossp = tf.reduce_mean(tf.square(self.f_pred - self.ftrain))
        
        if scheme == 'VPINNs':
            self.loss  = 10*self.lossb + self.lossv 
        if scheme == 'PINNs':
            self.loss  = 10*self.lossb + self.lossp 
        
        self.optimizer_Adam = tf.train.AdamOptimizer(0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
#        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        self.sess = tf.Session()
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
#            if it % 1 == 0:
#                loss_value = self.sess.run(self.loss, tf_dict)
#                u_pred     = self.sess.run(self.u_test, tf_dict)
#                u_pred_his.append(u_pred)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Loss: %.3e, Time: %.2f'])
                print(str_print % (it, loss_value, elapsed))
                start_time = time.time()

    def predict(self):
        u_pred = self.sess.run(self.u_test, {self.x_test: self.xtest, self.y_test: self.ytest})
        return u_pred


#%%
###############################################################################
# =============================================================================
#                               Main
# =============================================================================    
if __name__ == "__main__":     

    '''
    Hyper-parameters: 
        scheme     = is either 'PINNs' or 'VPINNs'
        Net_layer  = the structure of fully connected network
        var_form   = the form of the variational formulation used in VPINNs
                     0, 1, 2: no, once, twice integration-by-parts
        N_el_x, N_el_y     = number of elements in x and y direction
        N_test_x, N_test_y = number of test functions in x and y direction
        N_quad     = number of quadrature points in each direction in each element
        N_bound    = number of boundary points in the boundary loss
        N_residual = number of residual points in PINNs
    '''
    scheme = 'VPINNs'
    Net_layer = [2] + [5] * 3 + [1]
    var_form  = 1
    N_el_x = 4
    N_el_y = 4
    N_test_x = N_el_x*[5]
    N_test_y = N_el_y*[5]
    N_quad = 10
    N_bound = 80
    N_residual = 100    
    

    ###########################################################################
    def Test_fcn_x(n,x):
       test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
       return test
    def Test_fcn_y(n,y):
       test  = Jacobi(n+1,0,0,y) - Jacobi(n-1,0,0,y)
       return test

    ###########################################################################    
    omegax = 2*np.pi
    omegay = 2*np.pi
    r1 = 10
    def u_ext(x, y):
        utemp = (0.1*np.sin(omegax*x) + np.tanh(r1*x)) * np.sin(omegay*(y))
        return utemp

    def f_ext(x,y):
        gtemp = (-0.1*(omegax**2)*np.sin(omegax*x) - (2*r1**2)*(np.tanh(r1*x))/((np.cosh(r1*x))**2))*np.sin(omegay*(y))\
                +(0.1*np.sin(omegax*x) + np.tanh(r1*x)) * (-omegay**2 * np.sin(omegay*(y)) )
        return gtemp
    
    ###########################################################################
    # Boundary points
    x_up = 2*lhs(1,N_bound)-1
    y_up = np.empty(len(x_up))[:,None]
    y_up.fill(1)
    b_up = np.empty(len(x_up))[:,None]
    b_up = u_ext(x_up, y_up)
    x_up_train = np.hstack((x_up, y_up))
    u_up_train = b_up

    x_lo = 2*lhs(1,N_bound)-1
    y_lo = np.empty(len(x_lo))[:,None]
    y_lo.fill(-1)
    b_lo = np.empty(len(x_lo))[:,None]
    b_lo = u_ext(x_lo, y_lo)
    x_lo_train = np.hstack((x_lo, y_lo))
    u_lo_train = b_lo

    y_ri = 2*lhs(1,N_bound)-1
    x_ri = np.empty(len(y_ri))[:,None]
    x_ri.fill(1)
    b_ri = np.empty(len(y_ri))[:,None]
    b_ri = u_ext(x_ri, y_ri)
    x_ri_train = np.hstack((x_ri, y_ri))
    u_ri_train = b_ri    

    y_le = 2*lhs(1,N_bound)-1
    x_le = np.empty(len(y_le))[:,None]
    x_le.fill(-1)
    b_le = np.empty(len(y_le))[:,None]
    b_le = u_ext(x_le, y_le)
    x_le_train = np.hstack((x_le, y_le))
    u_le_train = b_le    

    X_u_train = np.concatenate((x_up_train, x_lo_train, x_ri_train, x_le_train))
    u_train = np.concatenate((u_up_train, u_lo_train, u_ri_train, u_le_train))

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
    # Construction of RHS for VPINNs
    NE_x, NE_y = N_el_x, N_el_y
    [x_l, x_r] = [-1, 1]
    [y_l, y_u] = [-1, 1]
    delta_x    = (x_r - x_l)/NE_x
    delta_y    = (y_u - y_l)/NE_y
    grid_x     = np.asarray([ x_l + i*delta_x for i in range(NE_x+1)])
    grid_y     = np.asarray([ y_l + i*delta_y for i in range(NE_y+1)])

#    N_testfcn_total = [(len(grid_x)-1)*[N_test_x], (len(grid_y)-1)*[N_test_y]]
    N_testfcn_total = [N_test_x, N_test_y]
 
    #+++++++++++++++++++
    x_quad  = XY_quad_train[:,0:1]
    y_quad  = XY_quad_train[:,1:2]
    w_quad  = WXY_quad_train
    U_ext_total = []
    F_ext_total = []
    for ex in range(NE_x):
        for ey in range(NE_y):
            Ntest_elementx  = N_testfcn_total[0][ex]
            Ntest_elementy  = N_testfcn_total[1][ey]
            
            x_quad_element = grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(x_quad+1)
            y_quad_element = grid_y[ey] + (grid_y[ey+1]-grid_y[ey])/2*(y_quad+1)
            jacobian       = ((grid_x[ex+1]-grid_x[ex])/2)*((grid_y[ey+1]-grid_y[ey])/2)
            
            testx_quad_element = np.asarray([ Test_fcn_x(n,x_quad)  for n in range(1, Ntest_elementx+1)])
            testy_quad_element = np.asarray([ Test_fcn_y(n,y_quad)  for n in range(1, Ntest_elementy+1)])
    
            u_quad_element = u_ext(x_quad_element, y_quad_element)
            f_quad_element = f_ext(x_quad_element, y_quad_element)
            
            U_ext_element = np.asarray([[jacobian*np.sum(\
                            w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*u_quad_element) \
                            for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
    
            F_ext_element = np.asarray([[jacobian*np.sum(\
                            w_quad[:,0:1]*testx_quad_element[r]*w_quad[:,1:2]*testy_quad_element[k]*f_quad_element) \
                            for r in range(Ntest_elementx)] for k in range(Ntest_elementy)])
            
            U_ext_total.append(U_ext_element)
    
            F_ext_total.append(F_ext_element)
    
#    U_ext_total = np.reshape(U_ext_total, [NE_x, NE_y, N_test_y, N_test_x])
    F_ext_total = np.reshape(F_ext_total, [NE_x, NE_y, N_test_y[0], N_test_x[0]])
    
    ###########################################################################
    # Test points
    delta_test = 0.01
    xtest = np.arange(x_l, x_r + delta_test, delta_test)
    ytest = np.arange(y_l, y_u + delta_test, delta_test)
    data_temp = np.asarray([[ [xtest[i],ytest[j],u_ext(xtest[i],ytest[j])] for i in range(len(xtest))] for j in range(len(ytest))])
    Xtest = data_temp.flatten()[0::3]
    Ytest = data_temp.flatten()[1::3]
    Exact = data_temp.flatten()[2::3]
    X_test = np.hstack((Xtest[:,None],Ytest[:,None]))
    u_test = Exact[:,None]


    ###########################################################################
    model = VPINN(X_u_train, u_train, X_f_train, f_train, XY_quad_train, WXY_quad_train,\
                  U_ext_total, F_ext_total, grid_x, grid_y, N_testfcn_total, X_test, u_test, Net_layer)
    
    u_pred_his, loss_his = [], []
    model.train(10000 + 1)
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
    fig.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_loss','.pdf']))
    
    ###########################################################################
    x_train_plot, y_train_plot = zip(*X_u_train)
    x_f_plot, y_f_plot = zip(*X_f_train)
    fig, ax = plt.subplots(1)
    if scheme == 'VPINNs':
        plt.scatter(x_train_plot, y_train_plot, color='red')
        for xc in grid_x:
            plt.axvline(x=xc, ymin=0.045, ymax=0.954, linewidth=1.5)
        for yc in grid_y:
            plt.axhline(y=yc, xmin=0.045, xmax=0.954, linewidth=1.5)
    if scheme == 'PINNs':
        plt.scatter(x_train_plot, y_train_plot, color='red')
        plt.scatter(x_f_plot,y_f_plot)
        plt.axhline(-1, linewidth=1, linestyle='--', color='k')
        plt.axhline(1, linewidth=1, linestyle='--', color='k')
        plt.axvline(-1, linewidth=1, linestyle='--', color='k')
        plt.axvline(1, linewidth=1, linestyle='--', color='k')
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.xlabel('$x$', fontsize = fontsize)
    plt.ylabel('$y$', fontsize = fontsize)
    #ax.set_aspect(1)
    ax.locator_params(nbins=5)
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Domain','.pdf']))
    
    ###########################################################################
    x_test_plot = np.asarray(np.split(X_test[:,0:1].flatten(),len(ytest)))
    y_test_plot = np.asarray(np.split(X_test[:,1:2].flatten(),len(ytest)))
    u_test_plot = np.asarray(np.split(u_test.flatten(),len(ytest)))
    u_pred_plot = np.asarray(np.split(u_pred.flatten(),len(ytest)))
    
    
    fontsize = 32
    labelsize = 26
    fig_ext, ax_ext = plt.subplots(constrained_layout=True)
    CS_ext = ax_ext.contourf(x_test_plot, y_test_plot, u_test_plot, 100, cmap='jet', origin='lower')
    cbar = fig_ext.colorbar(CS_ext, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_ext.locator_params(nbins=8)
    ax_ext.set_xlabel('$x$' , fontsize = fontsize)
    ax_ext.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_ext.set_aspect(1)
    #fig.tight_layout()
    fig_ext.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Exact','.png']))
    
    
    
    fig_pred, ax_pred = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred.contourf(x_test_plot, y_test_plot, u_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred.colorbar(CS_pred, shrink=0.67)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred.locator_params(nbins=8)
    ax_pred.set_xlabel('$x$' , fontsize = fontsize)
    ax_pred.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_pred.set_aspect(1)
    #fig.tight_layout()
    fig_pred.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_Predict','.png']))
    
    
    
    fig_err, ax_err = plt.subplots(constrained_layout=True)
    CS_err = ax_err.contourf(x_test_plot, y_test_plot, abs(u_test_plot - u_pred_plot), 100, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err, shrink=0.65, format="%.4f")
    cbar.ax.tick_params(labelsize = labelsize)
    ax_err.locator_params(nbins=8)
    ax_err.set_xlabel('$x$' , fontsize = fontsize)
    ax_err.set_ylabel('$y$' , fontsize = fontsize)
    plt.tick_params( labelsize = labelsize)
    ax_err.set_aspect(1)
    #fig.tight_layout()
    fig_err.set_size_inches(w=11,h=11)
    plt.savefig(''.join(['Poisson2D_',scheme,'_PntErr','.png']))
    
    
