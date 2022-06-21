
"""
@Title:
    hp-VPINNs: A General Framework For Solving PDEs
    Application to Advection Diffusion Eqn
@author: 
    Ehsan Kharazmi
    Division of Applied Mathematics
    Brown University
    ehsan_kharazmi@brown.edu
"""


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from pyDOE import lhs
import scipy.io
from scipy.special import legendre
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time

np.random.seed(1234)
tf.set_random_seed(1234)
###############################################################################
###############################################################################
PATH = './'
case = 'hpPINN_ADE_Iden'


LR = 0.001
Opt_Niter = 1500 + 1
Opt_tresh = 2e-11
var_form  = 0


gamma = 0.1
epsilon = gamma/(np.pi)
V = 1.0
T = 1

Net_layer = [2] + [5] * 3 + [1] 
N_el_x = 1
N_el_t = 1
N_test_x = N_el_x*[5]
N_test_t = N_el_t*[5]
N_quad = 10
N_bound = 80

test_Number = ''
#%%
###############################################################################
###############################################################################
class VPINN:
    # Initialize the class
    def __init__(self, XT_u_train, u_train, XT_f_train, XT_quad, W_quad,\
                 T_quad, WT_quad, grid_x, grid_t, N_testfcn, XT_test, u_test, layers, lb, ub):
    
        self.epsilon = tf.Variable(1*tf.ones([1], dtype=tf.float64), dtype=tf.float64)

        self.lb = lb
        self.ub = ub
        self.x    = XT_u_train[:,0:1]
        self.t    = XT_u_train[:,1:2]
        self.u    = u_train
        self.x_f    = XT_f_train[:,0:1]
        self.t_f    = XT_f_train[:,1:2]
        self.xquad  = XT_quad[:,0:1]
        self.tquad  = XT_quad[:,1:2]
        self.wquad  = W_quad

        self.tquad_1d = T_quad[:,None]
        self.xquad_1d = self.tquad_1d
        self.wquad_1d = WT_quad[:,None]
        
        self.xtest  = XT_test[:,0:1]
        self.ttest  = XT_test[:,1:2]
        self.utest  = u_test
        
        self.Nelementx = np.size(N_testfcn[0])
        self.Nelementt = np.size(N_testfcn[1])
        
       
        self.x_tf   = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.t_tf   = tf.placeholder(tf.float64, shape=[None, self.t.shape[1]])
        self.u_tf   = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]]) 

        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])

        self.x_test = tf.placeholder(tf.float64, shape=[None, self.xtest.shape[1]])
        self.t_test = tf.placeholder(tf.float64, shape=[None, self.ttest.shape[1]])
        self.x_quad = tf.placeholder(tf.float64, shape=[None, self.xquad.shape[1]])
        self.t_quad = tf.placeholder(tf.float64, shape=[None, self.tquad.shape[1]])

        self.weights, self.biases, self.a = self.initialize_NN(layers)

      
        self.u_NN_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        self.u_NN_test = self.net_u(self.x_test, self.t_test)

    
        self.varloss_total = 0

        for ex in range(self.Nelementx):
            for et in range(self.Nelementt):
                Ntest_elementx = N_testfcn[0][ex]
                Ntest_elementt = N_testfcn[1][et]

                jacobian       = (grid_t[et+1]-grid_t[et])/2*(grid_x[ex+1]-grid_x[ex])/2
                jacobian_x     = (grid_x[ex+1]-grid_x[ex])/2
                jacobian_t     = (grid_t[et+1]-grid_t[et])/2

                ##### 2D Integral evaluations  #####
                x_quad_element = tf.constant(grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(self.xquad+1), dtype=tf.float64)
                t_quad_element = tf.constant(grid_t[et] + (grid_t[et+1]-grid_t[et])/2*(self.tquad+1), dtype=tf.float64)

                u_NN_quad_element = self.net_u(x_quad_element, t_quad_element)
                d1xu_NN_quad_element, d2xu_NN_quad_element = self.net_dxu(x_quad_element, t_quad_element)
                d1tu_NN_quad_element = self.net_dtu(x_quad_element, t_quad_element)
                
                testx_quad_element = self.Test_fcn(Ntest_elementx, self.xquad)
                d1testx_quad_element, d2testx_quad_element = self.dTest_fcn(Ntest_elementx, self.xquad)
                testt_quad_element = self.Test_fcn(Ntest_elementt, self.tquad)
                d1testt_quad_element, d2testt_quad_element = self.dTest_fcn(Ntest_elementt, self.tquad)

                ##### 1D Integral evaluations for boundary terms in integration-by-parts #####
                t_quad_1d_element = tf.constant(grid_t[et] + (grid_t[et+1]-grid_t[et])/2*(self.tquad_1d+1), dtype=tf.float64)
                x_quad_1d_element = tf.constant(grid_x[ex] + (grid_x[ex+1]-grid_x[ex])/2*(self.xquad_1d+1), dtype=tf.float64)
#                x_b_element    = tf.constant(np.array([[grid_x[ex]], [grid_x[ex+1]]]))
                x_bl_element    = tf.constant(grid_x[ex], dtype=tf.float64, shape=t_quad_1d_element.shape) 
                x_br_element    = tf.constant(grid_x[ex+1], dtype=tf.float64, shape=t_quad_1d_element.shape) 
                t_tl_element    = tf.constant(grid_t[et], dtype=tf.float64, shape=x_quad_1d_element.shape) 
                t_tr_element    = tf.constant(grid_t[et+1], dtype=tf.float64, shape=x_quad_1d_element.shape) 

                u_NN_br_element = self.net_u(x_br_element, t_quad_1d_element)
                d1u_NN_br_element, d2u_NN_br_element = self.net_dxu(x_br_element, t_quad_1d_element)
                u_NN_bl_element = self.net_u(x_bl_element, t_quad_1d_element)
                d1u_NN_bl_element, d2u_NN_bl_element = self.net_dxu(x_bl_element, t_quad_1d_element)
                
                u_NN_tr_element = self.net_u(x_quad_1d_element, t_tr_element)
                u_NN_tl_element = self.net_u(x_quad_1d_element, t_tl_element)
                
                testx_bound_element = self.Test_fcn(Ntest_elementx, np.array([[-1],[1]]))
                d1testx_bound_element, d2testx_bounda_element = self.dTest_fcn(Ntest_elementx, np.array([[-1],[1]]))
                testt_quad_1d_element = self.Test_fcn(Ntest_elementt, self.tquad_1d)
                
                testt_bound_element = self.Test_fcn(Ntest_elementt, np.array([[-1],[1]]))
                testx_quad_1d_element = self.Test_fcn(Ntest_elementx, self.xquad_1d)
                

                integrand_part1 = d1tu_NN_quad_element 
                integrand_part1_VF22 = 0               

                
                if var_form == 0:
                    U_NN_element = tf.convert_to_tensor([[\
                                    jacobian*tf.reduce_sum(\
                                    self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testt_quad_element[k]\
                                    *(d1tu_NN_quad_element + V*d1xu_NN_quad_element - self.epsilon*d2xu_NN_quad_element))\
#                                    *(d1tu_NN_quad_element - self.epsilon*d2xu_NN_quad_element))\
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementt)], dtype= tf.float64)
                        
                if var_form == 1:

                    U_NN_element = tf.convert_to_tensor([[\
                         jacobian*tf.reduce_sum(self.wquad[:,0:1]*testx_quad_element[r]*self.wquad[:,1:2]*testt_quad_element[k]*(d1tu_NN_quad_element + V*d1xu_NN_quad_element))\
                       + self.epsilon*jacobian/jacobian_x*tf.reduce_sum(self.wquad[:,0:1]*d1testx_quad_element[r]*self.wquad[:,1:2]*testt_quad_element[k]*d1xu_NN_quad_element)\
                                    for r in range(Ntest_elementx)] for k in range(Ntest_elementt)], dtype= tf.float64)
                    



   
                Res_NN_element = tf.reshape(U_NN_element , [1,-1])
                loss_element = tf.reduce_mean(tf.square(Res_NN_element))
                self.varloss_total = self.varloss_total + loss_element
      
        self.lossb = 10*tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        self.lossv = self.varloss_total
        self.lossp = tf.reduce_mean(tf.square(self.f_pred))
        self.loss  = self.lossb + self.lossv #+ self.varloss_b_total#+ 0.01*loss_l2
                   


        self.LR = LR
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
#        self.sess = tf.Session()
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
        # H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):  
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases,  self.a)
        return u

    def net_dxu(self, x, t):
        u   = self.net_u(x, t)
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        return d1xu, d2xu
    
    def net_dtu(self, x, t):
        u   = self.net_u(x, t)
        d1tu = tf.gradients(u, t)[0]
        return d1tu

    def net_f(self, x, t):
        u = self.net_u(x,t)
        d1tu = tf.gradients(u, t)[0]
        d1xu = tf.gradients(u, x)[0]
        d2xu = tf.gradients(d1xu, x)[0]
        f = d1tu + V*d1xu - self.epsilon*d2xu
        return f



    def Test_fcn(self, N_test,x):
        test_total = []
        for n in range(1,N_test+1):
            test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
            test_total.append(test)
        return np.asarray(test_total)

## Legendre Test
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


    def callback(self, lossv, lossb):
        print('Lossv: %e, Lossb: %e' % (lossv, lossb))
        
###############################################################################
    def train(self, nIter, tresh):
        
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,\
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,\
                   self.x_quad: self.xquad, self.t_quad: self.tquad,\
                   self.x_test: self.xtest, self.t_test: self.ttest}
        
        total_time_train = 0
        min_loss         = 1e16
        start_time       = time.time()
        total_records    = []
        error_records    = []
        u_records_iterhis   = []

        for it in range(nIter):
            
            start_time_train = time.time()
            self.sess.run(self.train_op_Adam, tf_dict)
            elapsed_time_train = time.time() - start_time_train
            total_time_train = total_time_train + elapsed_time_train            
            
 
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_valueb= self.sess.run(self.lossb, tf_dict)
                loss_valuev= self.sess.run(self.lossv, tf_dict)
                loss_valuep= 1 
                epsilon_value = self.sess.run(self.epsilon, tf_dict)
                a_value = 1 
                total_records.append(np.array([it, loss_value, epsilon_value, a_value]))
                
                if loss_value < tresh:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    break

                if it > 0.9*nIter and loss_value < min_loss:
                    min_loss      = loss_value
                    u_pred     = self.sess.run(self.u_NN_test, tf_dict)
                    u_records     = u_pred
               
            if it % 100 == 0:
                elapsed = time.time() - start_time
                str_print = ''.join(['It: %d, Lossv: %.3e, Lossp: %.3e, Lossb: %.3e, Time: %.2f, TrTime: %.4f, epsilon: %.4f'])
                print(str_print % (it, loss_valuev, loss_valuep, loss_valueb, elapsed, elapsed_time_train, epsilon_value))
                start_time = time.time()
                
        error_u = 1
        error_records = [loss_value, error_u]

        return error_records, total_records, u_records, u_records_iterhis, total_time_train



###############################################################################

if __name__ == "__main__": 
    
    ###########################################################################    

    def u_initial(x, t):
        utemp = - np.sin(np.pi*x)
        return utemp

    
    ###########################################################################
    NPb = N_bound
    t_up = T*lhs(1,NPb)
    x_up = np.empty(len(t_up))[:,None]
    x_up.fill(1)
    b_up = np.empty(len(t_up))[:,None]
    b_up.fill(0.0)
    x_up_train = np.hstack((x_up, t_up))
    u_up_train = b_up
    
    t_lo = T*lhs(1,NPb)
    x_lo = np.empty(len(t_lo))[:,None]
    x_lo.fill(-1)
    b_lo = np.empty(len(t_lo))[:,None]
    b_lo.fill(0.0)
    x_lo_train = np.hstack((x_lo, t_lo))
    u_lo_train = b_lo

    x_in = 2*lhs(1,NPb)-1
    t_in = np.empty(len(x_in))[:,None]
    t_in.fill(0)
    b_in = np.empty(len(x_in))[:,None]
    b_in = u_initial(x_in, t_in)
    x_in_train = np.hstack((x_in, t_in))
    u_in_train = b_in


    XT_u_train = np.concatenate((x_up_train, x_lo_train, x_in_train))
    u_train = np.concatenate((u_up_train, u_lo_train, u_in_train))

    ###########################################################################
    NPf = 500
    grid_pt = lhs(2,NPf)
    xf = 2*grid_pt[:,0]-1
    timef = T*grid_pt[:,1]
    XT_f_train = np.hstack((xf[:,None],timef[:,None]))

    ###########################################################################
    # Quadrature points
    [X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
    T_quad, WT_quad   = (X_quad, WX_quad)
    xx, tt            = np.meshgrid(X_quad,  T_quad)
    wxx, wtt          = np.meshgrid(WX_quad, WT_quad)
    XT_quad_train     = np.hstack((xx.flatten()[:,None],  tt.flatten()[:,None]))
    WXT_quad_train    = np.hstack((wxx.flatten()[:,None], wtt.flatten()[:,None]))

    ###########################################################################
    NE_x, NE_t = N_el_x, N_el_t
    [x_l, x_r] = [-1, 1]
    [t_i, t_f] = [0, T]
    delta_x    = (x_r - x_l)/NE_x
    delta_t    = (t_f - t_i)/NE_t
    grid_x     = np.asarray([ x_l + i*delta_x for i in range(NE_x+1)])
    grid_t     = np.asarray([ t_i + i*delta_t for i in range(NE_t+1)])

    N_testfcn_total = [N_test_x, N_test_t]

 
#%%    
    ###########################################################################
    def u_ext(x, t, trunc = 800):
        """
        Function to compute the analytical solution as a Fourier series expansion.
        Inputs:
            x: column vector of locations
            t: column vector of times
            trunc: truncation number of Fourier bases
        """

        # Series index:
        p = np.arange(0, trunc+1.0)
        p = np.reshape(p, [1, trunc+1])
        
        D = epsilon
        c0 = 16*np.pi**2*D**3*V*np.exp(V/D/2*(x-V*t/2))                           # constant
        
        c1_n = (-1)**p*2*p*np.sin(p*np.pi*x)*np.exp(-D*p**2*np.pi**2*t)           # numerator of first component
        c1_d = V**4 + 8*(V*np.pi*D)**2*(p**2+1) + 16*(np.pi*D)**4*(p**2-1)**2     # denominator of first component
        c1 = np.sinh(V/D/2)*np.sum(c1_n/c1_d, axis=-1, keepdims=True)             # first component of the solution
        
        c2_n = (-1)**p*(2*p+1)*np.cos((p+0.5)*np.pi*x)*np.exp(-D*(2*p+1)**2*np.pi**2*t/4)
        c2_d = V**4 + (V*np.pi*D)**2*(8*p**2+8*p+10) + (np.pi*D)**4*(4*p**2+4*p-3)**2
        c2 = np.cosh(V/D/2)*np.sum(c2_n/c2_d, axis=-1, keepdims=True)       # second component of the solution
        
        c = c0*(c1+c2)
        
        if t==0:
            c = u_initial(x, t)
        
        return c


    delta_test = 0.01
    xtest = np.linspace(-1,1,256) 
    ttest = np.arange(0, T+delta_test, delta_test)
    data_temp = np.asarray([[ [xtest[i],ttest[j],u_ext(xtest[i],ttest[j])] for i in range(len(xtest))] for j in range(len(ttest))])
    ttest=ttest[:,None]
    xtest=xtest[:,None]
    Xtest = data_temp.flatten()[0::3]
    Ytest = data_temp.flatten()[1::3]
    Exact = data_temp.flatten()[2::3]
    XT_test = np.hstack((Xtest[:,None],Ytest[:,None]))
    u_test = Exact[:,None]

    lb = XT_test.min(0)
    ub = XT_test.max(0)

    #### Interior training points for inverse problem
    NPu_inter = 5
    x_inter_1 = np.empty(NPu_inter)[:,None]
    x_inter_1.fill(-0.5)
    t_inter_1 = T*lhs(1,NPu_inter)

    x_inter_2 = np.empty(NPu_inter)[:,None]
    x_inter_2.fill(0.0)
    t_inter_2 = T*lhs(1,NPu_inter)
    
    x_inter_3 = np.empty(NPu_inter)[:,None]
    x_inter_3.fill(0.5)
    t_inter_3 = T*lhs(1,NPu_inter)

    xu_inter = np.concatenate([x_inter_1, x_inter_2, x_inter_3])
    timeu_inter = np.concatenate([t_inter_1, t_inter_2, t_inter_3])
    XT_u_inter_train = np.hstack((xu_inter,timeu_inter))
    u_inter_train = np.asarray([ u_ext(XT_u_inter_train[i,0],XT_u_inter_train[i,1]) for i in range(XT_u_inter_train.shape[0])]).flatten()[:,None]

    XT_u_train = np.concatenate((x_up_train, x_lo_train, x_in_train, XT_u_inter_train))
    u_train = np.concatenate((u_up_train, u_lo_train, u_in_train, u_inter_train))


#%%
    ###########################################################################
    model = VPINN(XT_u_train, u_train, XT_f_train, XT_quad_train, WXT_quad_train,\
                  T_quad, WT_quad, grid_x, grid_t, N_testfcn_total, XT_test, u_test, Net_layer, lb, ub)
    
    total_record_total=[]
#%%
    error_record, total_record, u_record, u_records_iterhis, total_time_train\
    = model.train(Opt_Niter, Opt_tresh)
    total_record_total.append([total_record])
#%%

    
    
    with open(''.join([PATH,str(case), str(test_Number),'_record.mat']), 'wb') as f:
        scipy.io.savemat(f, {'x_test'      : XT_test})
        scipy.io.savemat(f, {'u_test'      : u_test})
        scipy.io.savemat(f, {'grid_x'      : grid_x})
        scipy.io.savemat(f, {'grid_t'      : grid_t})
        scipy.io.savemat(f, {'u_pred'      : u_record})
        scipy.io.savemat(f, {'u_pred_his'  : u_records_iterhis})
        scipy.io.savemat(f, {'total'       : total_record})
        scipy.io.savemat(f, {'total_time_train'   : total_time_train})
    
#%%

    # =============================================================================
    #     Loss
    # =============================================================================


    iteration = np.asarray(total_record)[:,0]
    loss_his  = np.asarray(total_record)[:,1]
    diff_his  = np.asarray(total_record)[:,2]


    fontsize = 24
    
    fig = plt.figure(2)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$loss \,\, values$', fontsize = fontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(iteration,loss_his)
    #legend = plt.legend(shadow=True, loc='upper center', fontsize=18, ncol = 3)
    plt.xticks(np.array([0, 50000, 100000, 150000]))
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=6) 
    plt.savefig(''.join([PATH,str(case),str(test_Number),'_loss','.pdf']))    


    fig = plt.figure(3)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize = fontsize)
    plt.ylabel('$diffusion \,\, coeff. \,\, (\kappa)$', fontsize = fontsize)
    plt.grid(True)
    plt.axhline(epsilon, linewidth=2, linestyle='--', color='r', label = 'exact $\kappa$')
    plt.plot(iteration,diff_his, linewidth=2, label = 'VPINN estimation')
    legend = plt.legend(shadow=True, loc='upper center', fontsize=18, ncol = 2)
    plt.xticks(np.array([0, 50000, 100000, 150000]))
    plt.tick_params( labelsize = 20)
    #fig.tight_layout()
    fig.set_size_inches(w=11,h=6)
    plt.savefig(''.join([PATH,str(case),str(test_Number),'_diffcoeff','.pdf']))    



#%%    

    # =============================================================================
    #     Training points and dimain discretization
    # =============================================================================
    
    x_train_plot, t_train_plot = zip(*XT_u_train)
    x_inter_train_plot, t_inter_train_plot = zip(*XT_u_inter_train)
    x_f_plot, t_f_plot = zip(*XT_f_train)
    x_q_plot, t_q_plot = zip(*XT_quad_train)
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    plt.scatter(t_train_plot, x_train_plot, color='red')
    
    plt.xlim([-0.02,1.02])
    plt.ylim([-1.1,1.1])
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.axhline(-1, linewidth=1, linestyle='--', color='k')
    plt.axhline(1, linewidth=1, linestyle='--', color='k')
    plt.axvline(0, linewidth=1, linestyle='--', color='k')
    plt.axvline(1, linewidth=1, linestyle='--', color='k')
    for xc in grid_x:
        plt.axhline(y=xc, xmin=0.02, xmax=0.98, linewidth=1.5)
    ax.set_aspect(1/5)
    fig.set_size_inches(w=20,h=4)
#    plt.savefig(''.join([PATH,str(case),str(test_Number),'_Domain','.pdf']))
    
    
    
    #%%
    # =============================================================================
    #     Fcn Plotting (3D)
    # =============================================================================
    
    x_test_plot = np.asarray(np.split(XT_test[:,0:1].flatten(),len(ttest)))
    y_test_plot = np.asarray(np.split(XT_test[:,1:2].flatten(),len(ttest)))
    z_test_plot = np.asarray(np.split(u_test.flatten(),len(ttest)))
    z_pred_plot = np.asarray(np.split(u_record.flatten(),len(ttest)))
#%%    

    fontsize = 32
    labelsize = 26
    
    
    fig_ext, ax_ext = plt.subplots(constrained_layout=True)
    CS_ext = ax_ext.contourf(y_test_plot, x_test_plot, z_test_plot, 100, cmap='jet', origin='lower')
    cbar = fig_ext.colorbar(CS_ext, shrink=1, ticks = np.arange(-1,1.1,0.4))
    cbar.ax.tick_params(labelsize = labelsize)
    plt.scatter(t_inter_train_plot, x_inter_train_plot, color = 'black', marker = 's', s =100)
    ax_ext.set_xlim(0, T)
    ax_ext.locator_params(nbins=5)
    ax_ext.set_aspect(1)
    ax_ext.set_xlabel('$t$', fontsize = fontsize)
    ax_ext.set_ylabel('$x$', fontsize = fontsize)
    ax_ext.tick_params(axis="x", labelsize = 18)
    ax_ext.tick_params(axis="y", labelsize = 18)
    plt.tick_params( labelsize = labelsize)
    ax_ext.axhline(y=-0.5, ls='--' ,color = 'black', linewidth=0.5)
    ax_ext.axhline(y=0   , ls='--' ,color = 'black', linewidth=0.5)
    ax_ext.axhline(y=0.5 , ls='--' ,color = 'black', linewidth=0.5)
    fig_ext.set_size_inches(w=11, h=11)
    plt.savefig(''.join([PATH,str(case),str(test_Number),'_FcnPlt_exact','.png']), dpi = 400)
#%% 

    fig_pred, ax_pred = plt.subplots(constrained_layout=True)
    CS_pred = ax_pred.contourf(y_test_plot, x_test_plot, z_pred_plot, 100, cmap='jet', origin='lower')
    cbar = fig_pred.colorbar(CS_pred, shrink=1, ticks = np.arange(-1,1.1,0.4))
    cbar.ax.tick_params(labelsize = labelsize)
    ax_pred.set_xlim(0, T)
    ax_pred.locator_params(nbins=5)
    ax_pred.set_aspect(1)
    ax_pred.set_xlabel('$t$', fontsize = fontsize)
    ax_pred.set_ylabel('$x$', fontsize = fontsize)
    ax_pred.tick_params(axis="x", labelsize = 18)
    ax_pred.tick_params(axis="y", labelsize = 18)
    plt.tick_params( labelsize = labelsize)
    fig_pred.set_size_inches(w=11, h=11)
    plt.savefig(''.join([PATH,str(case),str(test_Number),'_FcnPlt_predict','.png']), dpi = 400)
    
    

    fig_err, ax_err = plt.subplots(constrained_layout=True)
    CS_err = ax_err.contourf(y_test_plot, x_test_plot, abs(z_test_plot - z_pred_plot), \
                             levels = 100, cmap='jet', origin='lower')
    cbar = fig_err.colorbar(CS_err)
    cbar.ax.tick_params(labelsize = labelsize)
    ax_err.set_xlim(0, T)
    ax_err.locator_params(nbins=5)
    ax_err.set_aspect(1)
    for xc in grid_x:
        ax_err.axhline(y=xc, color = 'white', linewidth=1.5)
    for xc in grid_t:
        ax_err.axvline(x=xc, color = 'white', linewidth=1.5)
    ax_err.set_xlabel('$t$', fontsize = fontsize)
    ax_err.set_ylabel('$x$', fontsize = fontsize)
    ax_err.tick_params(axis="x", labelsize = 18)
    ax_err.tick_params(axis="y", labelsize = 18)
    plt.tick_params( labelsize = labelsize)
    fig_err.set_size_inches(w=11,h=11)
    plt.savefig(''.join([PATH,str(case),str(test_Number),'_PntErr','.png']), dpi = 400)
    
    
#%%
    
    # =============================================================================
    #     Fcn Plotting (slices)
    # =============================================================================
    #####################################################
    y_indices = np.arange(0, len(ttest), 20) #[ 0, 15, 30, 45, 60, 75] 
    plotgridcol = 3
    plotgridrow = int(len(y_indices)/plotgridcol)+1
    
    font = 14
    
    fig = plt.figure(3)
    gridspec.GridSpec(plotgridrow,plotgridcol)
    
    counter = 0
    for counterrow in range(plotgridrow):
        for countercol in range(plotgridcol):
            if counter < len(y_indices):    
                y_indx = y_indices[counter]
                L_inf_error = max(abs(z_test_plot[y_indx] - z_pred_plot[y_indx]))
                ax = plt.subplot2grid((plotgridrow,plotgridcol), (counterrow,countercol))
                ax.set_title(''.join(['$t = %.2f$, $|.|_{L^{\infty}} = %.3f$']) % (ttest[y_indices[counter],0], L_inf_error))
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.5, 1.5)
                ax.set_xticks([-1, -0.5, 0, 0.5, 1])
                ax.set_yticks([])
                for xc in grid_x:
                    ax.axvline(x=xc, linewidth=1.5, ls = '--')
                
                counter = counter + 1
                ax.plot(x_test_plot[y_indx], z_test_plot[y_indx], linewidth=2, color='red')
                ax.plot(x_test_plot[y_indx], z_pred_plot[y_indx], linewidth=3.5, color='black', ls = ':')


                
    
    fig.tight_layout()
    fig.set_size_inches(w= 10, h= 5)
    
    
    

