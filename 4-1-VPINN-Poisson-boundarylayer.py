
"""
Function Approximation

@Date: April 10 2019
@Author: Ehsan Kharazmi

"""
import sys
#sys.path.insert(0, '/users/ekharazm/Downloads')
sys.path.insert(0, 'C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Tools')
#sys.path.insert(0, '..\\..\\Tools')


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
#from matplotlib import gridspec
import matplotlib.gridspec as gridspec
from pyDOE import lhs
import scipy.io
from scipy.special import legendre
#from scipy.special import jacobi
#from scipy.special import gamma

#from GaussJacobiQuadRule_V2 import GaussJacobiWeights
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
import time

np.random.seed(1234)
tf.set_random_seed(1234)
###############################################################################
###############################################################################

PATH = '/users/ekharazm/Documents/Variational-PINN/'
PATH = ''.join([PATH,'VPINN-Poisson/Results/'])


#test_Number = '_testLegendre_NE3'
test_Number = '_testLegendre_NE3'
case = 'EVPINN_Poisson_Actsin'

LR = 0.001
Opt_Niter = 35000 + 1
Opt_tresh = 2e-32
var_form  = 1

#alpha = -1/2
#beta = -1/2

Net_layer = [1] + [20] * 4 + [1] 
N_test = 60
N_Quad = 80

PATH = ''.join([PATH,'SteepFcn/'])
test_Number =''.join([test_Number,'_L',str(len(Net_layer)-2),'N',str(Net_layer[1])])
#case_temp   =''.join([case, '_L',str(len(Net_layer)-2),'N',str(Net_layer[1])])
case_temp   =''.join([case, test_Number])


###############################################################################
###############################################################################
class PINN_WeakForm:
    # Initialize the class
    def __init__(self, X_u_train, u_train, X_bound, X_quad, W_quad, F_exact, U_exact_total, F_exact_total,\
                 grid, X_test, u_test, f_test, u_test_total, layers, X_f_train, f_train):

        self.x       = X_u_train
        self.u       = u_train
        self.xf      = X_f_train
        self.f      = f_train

        self.xb      = X_bound
        self.xquad   = X_quad
        self.wquad   = W_quad
        self.xtest   = X_test
        self.utest   = u_test
        self.ftest   = f_test
        self.utest_total = u_test_total
        self.grid_test   = grid
        
        self.U_ext_total = U_exact_total
        self.F_ext_total = F_exact_total
        self.F_ext       = F_exact
#        self.Nelement, self.Ntest, temp = np.asarray(np.shape(self.U_ext_total))
        self.Nelement = np.shape(self.U_ext_total)[0]
        self.N_test   = np.shape(self.U_ext_total[0])[0]
        
        
        self.x_tf   = tf.placeholder(tf.float64, shape=[None, self.x.shape[1]])
        self.u_tf   = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])

        self.xf_tf   = tf.placeholder(tf.float64, shape=[None, self.xf.shape[1]])
        self.f_tf   = tf.placeholder(tf.float64, shape=[None, self.f.shape[1]])

        self.x_test = tf.placeholder(tf.float64, shape=[None, self.xtest.shape[1]])
        self.x_quad = tf.placeholder(tf.float64, shape=[None, self.xquad.shape[1]])
        self.x_b    = tf.placeholder(tf.float64, shape=[None, self.xb.shape[1]])
      
        self.weights, self.biases, self.a = self.initialize_NN(layers)

        # NN output evaluated at different x
        self.u_NN_quad  = self.net_u(self.x_quad)
        self.d1u_NN_quad, self.d2u_NN_quad = self.net_du(self.x_quad)
        self.test_quad   = self.Test_fcn(self.N_test, self.xquad)
        self.d1test_quad, self.d2test_quad = self.dTest_fcn(self.N_test, self.xquad)
        
        self.u_NN_pred   = self.net_u(self.x_tf)
        self.u_NN_test   = self.net_u(self.x_test)

        self.f_pred = self.net_f(self.x_test)
        
        self.f_penalize = self.net_f(self.xf_tf)
        
        
        self.varloss_total = 0
        for e in range(self.Nelement):
            F_ext_element  = self.F_ext_total[e]
            Ntest_element  = np.shape(F_ext_element)[0]
            
            x_quad_element = tf.constant(grid[e] + (grid[e+1]-grid[e])/2*(self.xquad+1))
            x_b_element    = tf.constant(np.array([[grid[e]], [grid[e+1]]]))
            jacobian       = (grid[e+1]-grid[e])/2

            test_quad_element = self.Test_fcn(Ntest_element, self.xquad)
            d1test_quad_element, d2test_quad_element = self.dTest_fcn(Ntest_element, self.xquad)
            u_NN_quad_element = self.net_u(x_quad_element)
            d1u_NN_quad_element, d2u_NN_quad_element = self.net_du(x_quad_element)

            u_NN_bound_element = self.net_u(x_b_element)
            d1u_NN_bound_element, d2u_NN_bound_element = self.net_du(x_b_element)
            test_bound_element = self.Test_fcn(Ntest_element, np.array([[-1],[1]]))
            d1test_bound_element, d2test_bounda_element = self.dTest_fcn(Ntest_element, np.array([[-1],[1]]))


            if var_form == 0:
                U_NN_element = tf.reshape(tf.stack([jacobian*tf.reduce_sum(self.wquad*d2u_NN_quad_element*test_quad_element[i]) \
                                                   for i in range(Ntest_element)]),(-1,1))
            if var_form == 1:
                U_NN_element = tf.reshape(tf.stack([-tf.reduce_sum(self.wquad*d1u_NN_quad_element*d1test_quad_element[i]) \
#                                                    +tf.reduce_sum(d1u_NN_bound_element*np.array([-test_bound_element[i][0], test_bound_element[i][-1]]))  \
                                                   for i in range(Ntest_element)]),(-1,1))
            if var_form == 2:
                U_NN_element = tf.reshape(tf.stack([jacobian*tf.reduce_sum(self.wquad*u_NN_quad_element*d2test_quad_element[i]) \
                                                   - tf.reduce_sum(u_NN_bound_element*np.array([-d1test_bound_element[i][0], d1test_bound_element[i][-1]]))  \
                                                   + tf.reduce_sum(d1u_NN_bound_element*np.array([-test_bound_element[i][0], test_bound_element[i][-1]]))  \
                                                   for i in range(Ntest_element)]),(-1,1))
                

            Res_NN_element = U_NN_element - F_ext_element
            loss_element = tf.reduce_mean(tf.square(Res_NN_element))
            self.varloss_total = self.varloss_total + loss_element
        

        
        
        loss_l2 = tf.add_n([tf.nn.l2_loss(w_) for w_ in self.weights])
        self.lossb = tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        self.lossp = tf.reduce_mean(tf.square(self.f_tf - self.f_penalize))        
        self.lossv = self.varloss_total

#        self.loss  = self.lossb + self.lossv #+ 0.01*loss_l2
        self.loss  = 10*self.lossb + self.lossp #+ 0.01*loss_l2
#        self.loss  = self.lossb + self.lossv + self.lossp #+ 0.01*loss_l2
                   
#        self.loss =  self.varloss_total/np.linalg.norm(F_exact_total.flatten(),2) + tf.reduce_mean(tf.square(self.u_tf - self.u_NN_pred))
        
        self.LR = LR
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
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
#            b = self.xavier_init(size=[1, layers[l+1]])
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
#            H = tf.tanh(100*a*tf.add(tf.matmul(H, W), b)) 
#            H = tf.sin(100*a*tf.add(tf.matmul(H, W), b)) 
#            H = tf.nn.relu(100*a*tf.add(tf.matmul(H, W), b)) 
#            H = tf.tanh(tf.add(tf.matmul(H, W), b))
#            H = tf.sigmoid(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
            H = tf.sin(tf.add(tf.matmul(H, W), b))
#            H = tf.cos(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x):  
        u = self.neural_net(tf.concat([x],1), self.weights, self.biases, self.a)
        return u

    def net_du(self, x):
        u   = self.net_u(x)
        d1u = tf.gradients(u, x)[0]
        d2u = tf.gradients(d1u, x)[0]
        return d1u, d2u

    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_xx
        return f




    def Test_fcn(self, N_test,x):
        test_total = []
        for n in range(1,N_test+1):
#            test  = Jacobi(n+1,-1/2,-1/2,x)/Jacobi(n+1,-1/2,-1/2,1)
#            test  = Jacobi(n+1,-1/2,-1/2,x)/Jacobi(n+1,-1/2,-1/2,1) - Jacobi(n-1,-1/2,-1/2,x)/Jacobi(n-1,-1/2,-1/2,1)
            test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
#            test = np.sin(n * np.pi * x)
            test_total.append(test)
        return np.asarray(test_total)

#
## Chebyshev Test
#    def dTest_fcn(self, N_test,x):
#        d1test_total = []
#        d2test_total = []
#        for n in range(1,N_test+1):
#            d1test = ((n+1)/2)*Jacobi(n,1/2,1/2,x)/Jacobi(n+1,-1/2,-1/2,1) 
#            d2test = ((n+2)*(n+1)/(2*2))*Jacobi(n-1,3/2,3/2,x)/Jacobi(n+1,-1/2,-1/2,1)
#            d1test_total.append(d1test)
#            d2test_total.append(d2test)    
#        return np.asarray(d1test_total), np.asarray(d2test_total)


## Chebyshev Test
#    def dTest_fcn(self, N_test,x):
#        d1test_total = []
#        d2test_total = []
#        for n in range(1,N_test+1):
#            if n==1:
#                d1test = ((n+1)/2)*Jacobi(n,1/2,1/2,x)/Jacobi(n+1,-1/2,-1/2,1) 
#                d2test = ((n+2)*(n+1)/(2*2))*Jacobi(n-1,3/2,3/2,x)/Jacobi(n+1,-1/2,-1/2,1)
#                d1test_total.append(d1test)
#                d2test_total.append(d2test)
#            elif n==2:
#                d1test = ((n+1)/2)*Jacobi(n,1/2,1/2,x)/Jacobi(n+1,-1/2,-1/2,1) - ((n-1)/2)*Jacobi(n-2,1/2,1/2,x)/Jacobi(n-1,-1/2,-1/2,1)
#                d2test = ((n+2)*(n+1)/(2*2))*Jacobi(n-1,3/2,3/2,x)/Jacobi(n+1,-1/2,-1/2,1)
#                d1test_total.append(d1test)
#                d2test_total.append(d2test)    
#            else:
#                d1test = ((n+1)/2)*Jacobi(n,1/2,1/2,x)/Jacobi(n+1,-1/2,-1/2,1) - ((n-1)/2)*Jacobi(n-2,1/2,1/2,x)/Jacobi(n-1,-1/2,-1/2,1)
#                d2test = ((n+2)*(n+1)/(2*2))*Jacobi(n-1,3/2,3/2,x)/Jacobi(n+1,-1/2,-1/2,1) - ((n)*(n-1)/(2*2))*Jacobi(n-3,3/2,3/2,x)/Jacobi(n-1,-1/2,-1/2,1)
#                d1test_total.append(d1test)
#                d2test_total.append(d2test)    
#        return np.asarray(d1test_total), np.asarray(d2test_total)


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

#
#
#    def dTest_fcn(self, N_test,x):
#        d1test_total = []
#        d2test_total = []
#        for n in range(1,N_test+1):
#            d1test = n * np.pi * np.cos(n * np.pi * x)
#            d2test = - (n*np.pi)**2 * np.sin(n * np.pi * x)
#            d1test_total.append(d1test)
#            d2test_total.append(d2test)    
#        return np.asarray(d1test_total), np.asarray(d2test_total)




    def predict(self, grid):
        error_u_total = []
        u_pred_total = []
        for e in range(self.Nelement):
            utest_element = self.utest_total[e]
            x_test_element = grid[e] + (grid[e+1]-grid[e])/2*(self.xtest+1)
            u_pred_element = self.sess.run(self.u_NN_test, {self.x_test: x_test_element})
#            u_pred_element  = self.net_u(x_test_element)
            error_u_element = np.linalg.norm(utest_element - u_pred_element,2)/np.linalg.norm(utest_element,2)
            error_u_total.append(error_u_element)
            u_pred_total.append(u_pred_element)
        return u_pred_total, error_u_total
        

    def callback(self, loss):
        print('Loss: %e' % (loss))
        
###############################################################################
    def train(self, nIter, tresh):
        
        
        tf_dict = {self.x_tf: self.x, self.u_tf: self.u,\
                   self.x_quad: self.xquad, self.x_test: self.xtest,\
                   self.xf_tf: self.xf, self.f_tf: self.f}
        
        start_time       = time.time()
        min_loss         = 1e16
        no_success_count = 0
        success_count    = 0
        total_records    = []
        error_records    = []
        u_records_iterhis   = []
        u_records_in_elements_iterhis = []
        final_residue = []

        for it in range(nIter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
 
            if it % 10 == 0:
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_valueb= self.sess.run(self.lossb, tf_dict)
                loss_valuev= self.sess.run(self.lossv, tf_dict)
                loss_valuep= self.sess.run(self.lossp, tf_dict)
                u_pred     = self.sess.run(self.u_NN_test, tf_dict)
                error_u    = np.linalg.norm(self.utest - u_pred,2)/np.linalg.norm(self.utest,2)
                total_records.append(np.array([it, loss_value, error_u]))
                
                if loss_value < tresh:
                    print('It: %d, Loss: %.3e' % (it, loss_value))
                    break
                
                if loss_value < min_loss:
                    min_loss      = loss_value
                    error_records = [loss_value, error_u]
                    u_records     = u_pred
                    u_records_iterhis.append(u_records)
                    
                    u_records_in_elements, error_in_elements = self.predict(self.grid_test)
                    u_records_in_elements_iterhis.append(u_records_in_elements)
                    final_residue_temp = self.sess.run(self.f_pred, {self.x_test: self.xtest}) - self.ftest
                    final_residue.append(final_residue_temp)
                    success_count = success_count + 1
                else:
                    no_success_count = no_success_count + 1
                
            # Print
            if it % 100 == 0:
                elapsed = time.time() - start_time
                a_value = self.sess.run(self.a, tf_dict)
                str_print = 'It: %d, a_value: %.3e, Lossb: %.3e, Lossv: %.3e, Lossp: %.3e, Er_u: %.3e, Time: %.2f'
                print(str_print % (it, a_value, loss_valueb, loss_valuev, loss_valuep, error_u, elapsed))
                start_time = time.time()

#        final_residue = self.sess.run(self.f_pred, {self.x_f_tf: self.xtest}) - self.ftest
        return error_records, total_records, u_records, error_in_elements,\
               u_records_iterhis, u_records_in_elements_iterhis, final_residue

##################################################################################        
#    def predict(self, X_test):
#        test_dict = {self.x_test: X_test}
#        u_test = self.sess.run(self.u_NN_test, test_dict)
#        return u_test
#
#    def predict_proj(self):
#        test_dict = {self.x_quad: self.xquad}
#        U_NN = self.sess.run(self.U_NN_proj, test_dict)
#        return U_NN
################################################################################

#%%
# =============================================================================
#                               Main Loop
# =============================================================================
def main_loop(model_layers_inp, model_N_quad_inp, model_N_test_inp, train_Niter_inp, train_tresh_inp):

    layers     = model_layers_inp
    NQ         = model_N_quad_inp
    N_testfcn  = model_N_test_inp
    #++++++++++++++++++++++++++++    

    def Test_fcn(n,x):
#       test  = Jacobi(n+1,-1/2,-1/2,x)/Jacobi(n+1,-1/2,-1/2,1)
#       test  = Jacobi(n+1,-1/2,-1/2,x)/Jacobi(n+1,-1/2,-1/2,1) - Jacobi(n-1,-1/2,-1/2,x)/Jacobi(n-1,-1/2,-1/2,1)
       test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
#       test  = np.sin(n * np.pi * x)
       return test

    #++++++++++++++++++++++++++++    
#    def u_ext(x):
#        utemp = np.heaviside(-x,0)*2*np.sin(6*np.pi*x) + np.heaviside(x,0)*(6 + 1*np.exp(1.2*x)*np.sin(12*np.pi*x))
##        utemp = np.heaviside(-(x+0.1),0)*2*np.sin(4*np.pi*x) + np.heaviside(x+0.1,0)*(6 + 1*np.exp(1.2*x)*np.sin(12*np.pi*x))
##        utemp = (1-x**2) + np.sin(8*np.pi*x**2)
##        utemp = np.heaviside(-x,0)*2*np.exp(3*x)*np.sin(6*np.pi*x) + np.heaviside(x,0)*(4 + 2*np.exp(-3*x)*np.sin(14*np.pi*x))
##        utemp = np.heaviside(-x,0)*1.5*np.sin(6*np.pi*x) + np.heaviside(x,0)*(5 + 0.5*np.exp(1.2*x)*np.sin(14*np.pi*x))
##        utemp = np.heaviside(-x,0)*0.2*np.sin(6*np.pi*x) + np.heaviside(x,0)*(1 + (0.2+0.6*x)*np.cos(12*np.pi*x))
#        return utemp
    omega = 8*np.pi
    omega1, omega2, omega3, omega4, omega5, omega6, omega7, omega8\
    = [1*np.pi, 2*np.pi, 3*np.pi, 4*np.pi, 5*np.pi, 6*np.pi, 7*np.pi, 8*np.pi]
    amp = 1
    g = 1.2
    omega_l = 2*np.pi
    omega_h = 6*np.pi
    r = 200
    r1 = 80
#    def u_ext(x):
#        utemp = (1-x**2)*np.sin(omega*x)
#        return amp*utemp
#    def f_ext(x):
#        gtemp = -4*omega*x*np.cos(omega*x) - 2*np.sin(omega*x) - (omega**2)*(1-x**2)*np.sin(omega*x)
#        return amp*gtemp
    def u_ext(x):
#        utemp = np.sin(omega*x)
#        utemp = 1 - np.cosh(r/2*x)/np.cosh(r/2)
#        utemp = 0.1*np.sin(omega*x) + np.tanh(r1*x)
#        utemp = x + np.sin(omega1*x)/omega1 + np.sin(omega2*x)/omega2 + np.sin(omega3*x)/omega3 + np.sin(omega4*x)/omega4\
#               +np.sin(omega5*x)/omega5 + np.sin(omega6*x)/omega6 + 5*np.sin(omega7*x)/omega7 + np.sin(omega8*x)/omega8
#        utemp = (1-x**2) + np.sin(omega*x**2)
#        utemp = np.heaviside(-x,0)*2*np.sin(omega_l*x) + np.heaviside(x,0)*(np.exp(g*x)*np.sin(omega_h*x))
        utemp = 0.1*np.sin(5*np.pi*x) + np.exp((0.01-x-1)/(0.01))
        return amp*utemp

    def f_ext(x):
#        gtemp = - (omega**2)*np.sin(omega*x)
#        gtemp = -r**2/4*np.cosh(r/2*x)/np.cosh(r/2)
#        gtemp =  -0.1*(omega**2)*np.sin(omega*x) - (2*r1**2)*(np.tanh(r1*x))/((np.cosh(r1*x))**2)
#        gtemp = -omega1*np.sin(omega1*x) - omega2*np.sin(omega2*x) - omega3*np.sin(omega3*x) - omega4*np.sin(omega4*x)\
#                -omega5*np.sin(omega5*x) - omega6*np.sin(omega6*x) - 5*omega7*np.sin(omega7*x) - omega8*np.sin(omega8*x)
#        gtemp = -2 + (2*omega)*np.cos(omega*x**2) - ((2*omega*x)**2)*np.sin(omega*x**2)
#        gtemp = np.heaviside(-x,0)*(-2*(omega_l**2)*np.sin(omega_l*x)) \
#        + np.heaviside(x,0)*( g*np.exp(g*x)*(g*np.sin(omega_h*x) + omega_h*np.cos(omega_h*x)) + np.exp(g*x)*(g*omega_h*np.cos(omega_h*x) - (omega_h**2)*np.sin(omega_h*x)) )
        gtemp = -0.1*((5*np.pi)**2)*np.sin(5*np.pi*x) + (-1/0.01)**2*np.exp((0.01-x-1)/(0.01))
        return amp*gtemp
    
    #++++++++++++++++++++++++++++
        
    #+++++++++++++++++++
    NQ_u = NQ
    [x_quad, w_quad] = GaussLobattoJacobiWeights(NQ_u, 0, 0)
#    x_quad = np.asarray([np.cos((2*i-1)*np.pi/(2*NQ_u)) for i in range(1,NQ_u+1)])
#    w_quad = np.asarray([np.pi/NQ_u for i in range(1,NQ_u+1)])
    testfcn = np.asarray([ Test_fcn(n,x_quad)  for n in range(1, N_testfcn+1)])
    
    NE = 1
    [x_l, x_r] = [-1, 1]
    delta_x = (x_r - x_l)/NE
    grid = np.asarray([ x_l + i*delta_x for i in range(NE+1)])
    N_testfcn_total = np.array((len(grid)-1)*[N_testfcn])
 
#    grid = np.array([-1, -0.1, 0.1, 1])
##    grid = np.array([-1,0, 1])
#    NE = len(grid)-1
#    N_testfcn_total = np.array([N_testfcn,N_testfcn,N_testfcn])
    
    U_ext_total = []
    F_ext_total = []
    for e in range(NE):
        x_quad_element = grid[e] + (grid[e+1]-grid[e])/2*(x_quad+1)
        jacobian = (grid[e+1]-grid[e])/2
        N_testfcn_temp = N_testfcn_total[e]
        testfcn_element = np.asarray([ Test_fcn(n,x_quad)  for n in range(1, N_testfcn_temp+1)])

        u_quad_element = u_ext(x_quad_element)
        U_ext_element  = jacobian*np.asarray([sum(w_quad*u_quad_element*testfcn_element[i]) for i in range(N_testfcn_temp)])
        U_ext_element = U_ext_element[:,None]
        U_ext_total.append(U_ext_element)

        f_quad_element = f_ext(x_quad_element)
        F_ext_element  = jacobian*np.asarray([sum(w_quad*f_quad_element*testfcn_element[i]) for i in range(N_testfcn_temp)])
        F_ext_element = F_ext_element[:,None]
        F_ext_total.append(F_ext_element)
    
    U_ext_total = np.asarray(U_ext_total)
    F_ext_total = np.asarray(F_ext_total)


    testfcn = np.asarray([ Test_fcn(n,x_quad)  for n in range(1, N_testfcn+1)])
    f_quad  = f_ext(x_quad)
    F       = np.asarray([sum(w_quad*f_quad*testfcn[i]) for i in range(N_testfcn)])
    F       = F[:,None]

    #+++++++++++++++++++
    # Training points
    X_u_train = np.asarray([-1.0,1.0])[:,None]
#    X_u_train = np.asarray(-1/2*(2*lhs(100,1)-1))
#    np.random.seed(1)
#    X_u_train = np.append(X_u_train, -1/8*lhs(100,1)+1/16)[:,None]
#    X_u_train = np.append(X_u_train, 2*lhs(300,1)-1)[:,None]
    u_train   = u_ext(X_u_train)

    X_bound = np.asarray([-1.0,1.0])[:,None]
    
    Nf = 500
#    X_f_train = np.array([-0.5])[:,None] 
    X_f_train = (2*lhs(1,Nf)-1)
#    delta_train = 0.02
#    X_f_train   = np.arange(-1 + delta_train , 1 , delta_train)[:,None]
    f_train   = f_ext(X_f_train)

    #+++++++++++++++++++
    # Quadrature points
    [x_quad, w_quad] = GaussLobattoJacobiWeights(NQ, 0, 0)
#    x_quad = np.asarray([np.cos((2*i-1)*np.pi/(2*NQ)) for i in range(1,NQ+1)])
#    w_quad = np.asarray([np.pi/NQ for i in range(1,NQ+1)])

    X_quad_train = x_quad[:,None]
    W_quad_train = w_quad[:,None]

    #+++++++++++++++++++
    # Test point
    delta_test = 0.001
    xtest      = np.arange(-1 , 1 + delta_test , delta_test)
    data_temp  = np.asarray([ [xtest[i],u_ext(xtest[i])] for i in range(len(xtest))])
    X_test = data_temp.flatten()[0::2]
    u_test = data_temp.flatten()[1::2]
    X_test = X_test[:,None]
    u_test = u_test[:,None]
    f_test = f_ext(X_test)
#    f_test = f_ext(X_quad_train)

    u_test_total = []
    for e in range(NE):
        x_test_element = grid[e] + (grid[e+1]-grid[e])/2*(xtest+1)
        u_test_element = u_ext(x_test_element)
        u_test_element = u_test_element[:,None]
        u_test_total.append(u_test_element)

    ##+++++++++++++++++++
    # Model and Training
    model = PINN_WeakForm(X_u_train, u_train, X_bound, X_quad_train, W_quad_train, F,\
                          U_ext_total, F_ext_total, grid, X_test, u_test, f_test, u_test_total, layers, X_f_train, f_train)

    
    error_record, total_record, u_record, error_in_elements,\
    u_records_iterhis, u_records_in_elements_iterhis, final_residue\
    = model.train(train_Niter_inp, train_tresh_inp)
    
#    error_in_elements = model.predict(grid)

    return error_record, total_record, u_record, \
           X_test, u_test, X_quad_train, X_u_train, F, F_ext_total, grid, X_f_train,\
           error_in_elements, u_records_iterhis, u_records_in_elements_iterhis, final_residue, f_test
    

#%%
################################################################################
## =============================================================================
##     Runnign the main loop
## =============================================================================


error_record, total_record, u_record, X_test, u_test, X_quad_train, X_u_train, Fext,\
F_ext_total, grid, X_f_train, error_in_elements, u_records_iterhis, u_records_in_elements_iterhis, final_residue, f_test  \
= main_loop(Net_layer, N_Quad, N_test, Opt_Niter, Opt_tresh)
print('\n\n error in each elements: \n \n', np.array(error_in_elements), '\n\n')




#%%
# =============================================================================
#     Training/penalizing/quadrature Points Plotting
# =============================================================================    
x_quad_plot = X_quad_train
y_quad_plot = np.empty(len(x_quad_plot))
y_quad_plot.fill(1)

x_train_plot = X_u_train
y_train_plot = np.empty(len(x_train_plot))
y_train_plot.fill(1) 

x_f_plot = X_f_train
y_f_plot = np.empty(len(x_f_plot))
y_f_plot.fill(1)

fig = plt.figure(0)
gridspec.GridSpec(3,1)

plt.subplot2grid((3,1), (0,0))
plt.tight_layout()
plt.locator_params(axis='x', nbins=6)
plt.yticks([])
plt.title('$Quadrature \,\, Points$')
#plt.xlabel('$x$')
plt.axhline(1, linewidth=1, linestyle='-', color='red')
plt.axvline(-1, linewidth=1, linestyle='--', color='red')
plt.axvline(1, linewidth=1, linestyle='--', color='red')
plt.scatter(x_quad_plot,y_quad_plot, color='green')

plt.subplot2grid((3,1), (1,0))
plt.tight_layout()
plt.locator_params(axis='x', nbins=6)
plt.yticks([])
plt.title('$Training \,\, Points$')
#plt.xlabel('$x$')
plt.axhline(1, linewidth=1, linestyle='-', color='red')
plt.axvline(-1, linewidth=1, linestyle='--', color='red')
plt.axvline(1, linewidth=1, linestyle='--', color='red')
plt.scatter(x_train_plot,y_train_plot, color='blue')
#plt.scatter(x_f_train_plot,y_f_train_plot, color='black')


plt.subplot2grid((3,1), (2,0))
plt.tight_layout()
plt.locator_params(axis='x', nbins=6)
plt.yticks([])
plt.title('$Penalizing \,\, Points$')
#plt.xlabel('$x$')
plt.axhline(1, linewidth=1, linestyle='-', color='red')
plt.axvline(-1, linewidth=1, linestyle='--', color='red')
plt.axvline(1, linewidth=1, linestyle='--', color='red')
plt.scatter(x_f_plot,y_f_plot, color='black')


# fit subplots and save fig
fig.tight_layout()
fig.set_size_inches(w=10,h=7)
#plt.savefig(''.join([PATH,str(case),str(test_Number),'_FcnPlt','.pdf']))    
#plt.show()
#%%
# =============================================================================
#     VPINN:  Plotting
# =============================================================================

font = 20
pnt_skip = 18
u_pred_temp = u_record

fig = plt.figure(1)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.xlabel('$x$', fontsize = font)
plt.ylabel('$u_{NN}$', fontsize = font)
plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
for xc in grid[0:-1]:
    plt.axvline(x=xc, linewidth=2, ls = '--')
plt.axvline(x=1, linewidth=2, ls = '--', label = '$domain \,\, disc.$')
plt.plot(X_test, u_test, linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(X_test[0::pnt_skip], u_pred_temp[0::pnt_skip], 'k*', label='$VPINN$')

legend = plt.legend(shadow=True, loc='upper right', fontsize='small')
plt.tight_layout()
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-NE1-fcnplt.pdf')



fig = plt.figure(11)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.xlabel('$x$', fontsize = font)
plt.ylabel('$point-wise \,\, error$', fontsize = font)
plt.ylim((0,0.005))
plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
for xc in grid[0:-1]:
    plt.axvline(x=xc, linewidth=2, ls = '--')
plt.axvline(x=1, linewidth=2, ls = '--', label = '$domain \,\, disc.$')
plt.plot(X_test, abs(u_pred_temp - u_test), linewidth=1, color='k', label=''.join(['$exact$']))
#legend = plt.legend(shadow=True, loc='upper left', fontsize='small')
plt.tight_layout()
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-NE1-pnterror.pdf')


i = len(u_records_iterhis)-1
t = np.arange(len(X_test))
freq = np.fft.fftfreq(t.shape[-1])
sp = np.fft.fft(u_test.flatten())
u_pred_temp = u_record.flatten()  #u_records_iterhis[i].flatten()
sp2 = np.fft.fft(u_pred_temp)

fig = plt.figure(21)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.title('$low \,\, frequency \,\, index$', fontsize = font)
temp = 35
temp2 = 100
plt.plot(t[1:temp], abs(sp[1:temp]), linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(t[1:temp], abs(sp2[1:temp]), 'k--', marker='*', label='$VPINN$')
plt.ylim((-30,500))
legend = plt.legend(shadow=True, loc='upper right', fontsize='small')
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-NE1-FFTlow.pdf')


fig = plt.figure(22)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.title('$high \,\, frequency \,\, index$', fontsize = font)
temp = 35
temp2 = 100
plt.plot(t[temp+1:temp2], abs(sp[temp+1:temp2]), linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(t[temp+1:temp2], abs(sp2[temp+1:temp2]), 'k--', marker = '*', label='$VPINN$')
plt.ylim((5,27))
legend = plt.legend(shadow=True, loc='upper right', fontsize='small')
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-NE1-FFThigh.pdf')


#%%

# =============================================================================
#     PINN:  Plotting
# =============================================================================


font = 20
pnt_skip = 25
u_pred_temp = u_record

fig = plt.figure(1)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.xlabel('$x$', fontsize = font)
plt.ylabel('$u_{NN}$', fontsize = font)
plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
#for xc in grid[0:-1]:
#    plt.axvline(x=xc, linewidth=2, ls = '--')
#plt.axvline(x=1, linewidth=2, ls = '--', label = '$domain \,\, disc.$')
plt.plot(X_test, u_test, linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(X_test[0::pnt_skip], u_pred_temp[0::pnt_skip], 'kd', label='$PINN$')

legend = plt.legend(shadow=True, loc='upper left', fontsize='small')
plt.tight_layout()
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-PINN-fcnplt.pdf')

fig = plt.figure(11)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.xlabel('$x$', fontsize = font)
plt.ylabel('$point-wise \,\, error$', fontsize = font)
#plt.ylim((0,0.015))
plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
#for xc in grid[0:-1]:
#    plt.axvline(x=xc, linewidth=2, ls = '--')
#plt.axvline(x=1, linewidth=2, ls = '--', label = '$domain \,\, disc.$')
plt.plot(X_test, abs(u_pred_temp - u_test), linewidth=1, color='k', label=''.join(['$exact$']))
#legend = plt.legend(shadow=True, loc='upper left', fontsize='small')
plt.tight_layout()
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-PINN-pnterror.pdf')


i = len(u_records_iterhis)-1
t = np.arange(len(X_test))
freq = np.fft.fftfreq(t.shape[-1])
sp = np.fft.fft(u_test.flatten())
u_pred_temp = u_record.flatten()  #u_records_iterhis[i].flatten()
sp2 = np.fft.fft(u_pred_temp)

fig = plt.figure(21)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.title('$low \,\, frequency \,\, index$', fontsize = font)
temp = 35
temp2 = 100
plt.plot(t[1:temp], abs(sp[1:temp]), linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(t[1:temp], abs(sp2[1:temp]), 'k--', marker='d', label='$PINN$')
plt.ylim((-30,500))
legend = plt.legend(shadow=True, loc='upper right', fontsize='small')
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-PINN-FFTlow.pdf')


fig = plt.figure(22)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.title('$high \,\, frequency \,\, index$', fontsize = font)
temp = 35
temp2 = 100
plt.plot(t[temp+1:temp2], abs(sp[temp+1:temp2]), linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(t[temp+1:temp2], abs(sp2[temp+1:temp2]), 'k--', marker = 'd', label='$PINN$')
plt.ylim((5,27))
legend = plt.legend(shadow=True, loc='upper right', fontsize='small')
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-PINN-FFThigh.pdf')



#%%

# =============================================================================
#     Error Plotting
# =============================================================================

iteration = np.array(total_record)[:,0]
loss_his = np.array(total_record)[:,1]
error_his = np.array(total_record)[:,2]

fig = plt.figure(3)
gridspec.GridSpec(1,1)
ax = plt.subplot2grid((1,1), (0,0))
#plt.tight_layout()
#plt.locator_params(axis='x', nbins=6)
#plt.locator_params(axis='y', nbins=6)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
plt.xlabel('$iteration$')
plt.ylabel('$loss \,\, values$')
plt.yscale('log')
plt.grid(True)
plt.plot(iteration, loss_his)
# fit subplots and save fig
fig.tight_layout()
fig.set_size_inches(w=8,h=8)
#plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-NE2-loss.pdf')
#plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-NE3-loss.pdf')
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-PINN-loss.pdf')
#plt.savefig(''.join([PATH,str(case),str(test_Number),'_loss','.pdf']))    
#plt.show()


fig = plt.figure(4)
gridspec.GridSpec(1,1)
ax = plt.subplot2grid((1,1), (0,0))
#plt.tight_layout()
#plt.locator_params(axis='x', nbins=6)
#plt.locator_params(axis='y', nbins=6)
plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
plt.xlabel('$iteration$')
plt.ylabel('$error \,\, values$')
plt.yscale('log')
plt.grid(True)
plt.plot(iteration, error_his)

# fit subplots and save fig
fig.tight_layout()
fig.set_size_inches(w=8,h=8)
#plt.savefig(''.join([PATH,str(case),str(test_Number),'_loss','.pdf']))    
#plt.show()


#with open(''.join([PATH,str(case),'_error_record', str(test_Number),'.mat']), 'wb') as f:
#    scipy.io.savemat(f, {'error'       : error_record_total})
#    scipy.io.savemat(f, {'total'       : total_record_total})



#%%

# =============================================================================
#     VPINN vs  PINN
# =============================================================================


font = 20
#u_pred_VPINN = u_record
u_pred_PINN = u_record

#%%
pnt_skip = 15

fig = plt.figure(1)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.xlabel('$x$', fontsize = font)
plt.ylabel('$u_{NN}$', fontsize = font)
plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
for xc in grid[0:-1]:
    plt.axvline(x=xc, linewidth=2, ls = '--')
plt.axvline(x=1, linewidth=2, ls = '--', label = '$domain \,\, disc.$')
plt.plot(X_test, u_test, linewidth=2, color='red', label=''.join(['$exact$']))
plt.plot(X_test[0::pnt_skip], u_pred_PINN[0::pnt_skip], 'g^', label='$PINN$')
plt.plot(X_test[0::pnt_skip], u_pred_VPINN[0::pnt_skip], 'k*', label='$VPINN$')


legend = plt.legend(shadow=True, loc='upper right', fontsize='small')
plt.tight_layout()
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-compare-fcnplt.pdf')


pnt_skip = 15
fig = plt.figure(11)
plt.locator_params(axis='x', nbins=6)
plt.locator_params(axis='y', nbins=8)
plt.xlabel('$x$', fontsize = font)
plt.ylabel('$point-wise \,\, error$', fontsize = font)
#plt.ylim((0,0.005))
plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
for xc in grid[0:-1]:
    plt.axvline(x=xc, linewidth=2, ls = '--')
plt.axvline(x=1, linewidth=2, ls = '--')
plt.plot(X_test, abs(u_pred_VPINN - u_test), color = 'k', linewidth=1)
plt.plot(X_test[0::pnt_skip], abs(u_pred_VPINN[0::pnt_skip] - u_test[0::pnt_skip]), 'k*', linewidth=1, label=''.join(['$VPINN$']))
plt.plot(X_test, abs(u_pred_PINN - u_test), 'k', linewidth=1)
plt.plot(X_test[0::pnt_skip], abs(u_pred_PINN[0::pnt_skip] - u_test[0::pnt_skip]), 'g^', linewidth=1, label=''.join(['$PINN$']))
legend = plt.legend(shadow=True, loc='upper left', fontsize='small')
plt.tight_layout()
fig.set_size_inches(w=10,h=3)
plt.savefig('C:\\Users\\Ehsan\\Google Drive\\Uni_Brown\\Crunch-Group\\Codes\\Variational-PINN\\VPINN-Poisson-1d\\plots\\BoundaryLayer-compare-pnterror.pdf')














#%%





#%%
for i in range(0,len(u_records_iterhis), -2):
    
    fig = plt.figure(5)
    gridspec.GridSpec(3,2)

    ax = plt.subplot2grid((3,2), (0,0))
    plt.title('')
    plt.xlabel('$x$')
    plt.ylabel('$u_{NN}$')
    for xc in grid:
        plt.axvline(x=xc, linewidth=1.2)
    plt.plot(X_test,u_test, linewidth=2, color='red')
#    ax.yaxis.grid(True)
    #plt.tight_layout()
    u_pred_temp = u_records_iterhis[i]
    plt.plot(X_test,u_pred_temp, linewidth=1, color='black')
    ax.set(ylim=[-1.3, 1.3])
    
    ax = plt.subplot2grid((3,2), (1,0))
    plt.title('')
    plt.xlabel('$frequency \,\, index$')
    plt.ylabel('')
    t = np.arange(len(X_test))
    freq = np.fft.fftfreq(t.shape[-1])
    sp = np.fft.fft(u_test.flatten())
    u_pred_temp = u_records_iterhis[i].flatten()
    sp2 = np.fft.fft(u_pred_temp)
    temp = 35
    temp2 = 100
    plt.plot(t[1:temp], abs(sp[1:temp]), linewidth=2, color='red')
    plt.plot(t[1:temp], abs(sp2[1:temp]), linewidth=1, color='black')
    ax.set(ylim=[-20,500])
    
    ax = plt.subplot2grid((3,2), (2,0))
    plt.title('')
    plt.xlabel('$frequency \,\, index$')
    plt.ylabel('')
    t = np.arange(len(X_test))
    freq = np.fft.fftfreq(t.shape[-1])
    sp = np.fft.fft(u_test.flatten())
    u_pred_temp = u_records_iterhis[i].flatten()
    sp2 = np.fft.fft(u_pred_temp)
    temp = 35
    temp2 = 100
    plt.plot(t[temp+1:temp2], abs(sp[temp+1:temp2]), linewidth=2, color='red')
    plt.plot(t[temp+1:temp2], abs(sp2[temp+1:temp2]), linewidth=1, color='black')
    ax.set(ylim=[5,27])

    #++++++++++++++++++++++++++++++++++++++++
    
    ax = plt.subplot2grid((3,2), (0,1))
    plt.title('')
    plt.xlabel('$x$')
    plt.ylabel('$residue}$')
#    ax.yaxis.grid(True)
    #plt.tight_layout()
    plt.axhline(0, linewidth=2, color='red')
    res_temp = final_residue[i]
    plt.plot(X_test, res_temp, linewidth=1, color='black')
    ax.set(ylim=[-400,400])

    ax = plt.subplot2grid((3,2), (1,1))
    plt.title('')
    plt.xlabel('$x$')
    plt.ylabel('$point-wise \,\, error$')
    for xc in grid:
        plt.axvline(x=xc, linewidth=1.2)
#    ax.yaxis.grid(True)
    #plt.tight_layout()
    u_pred_temp = u_records_iterhis[i]
    plt.plot(X_test,u_pred_temp-u_test, linewidth=1, color='black')
    ax.set(ylim=[-1.3, 1.3])

    ax = plt.subplot2grid((3,2), (2,1))
    plt.title('')
    plt.xlabel('$x$')
    plt.ylabel('$point-wise \,\, error$')
    for xc in grid:
        plt.axvline(x=xc, linewidth=1.2)
#    ax.yaxis.grid(True)
    #plt.tight_layout()
    u_pred_temp = u_records_iterhis[i]
    plt.plot(X_test,u_pred_temp-u_test, linewidth=1, color='black')
    ax.set(ylim=[-0.1, 0.1])

    #++++++++++++++++++++++++++++++++++++++++
    
#    fig.tight_layout()
    fig.set_size_inches(w=22,h=12)
#    plt.savefig(''.join([PATH, 'figplotsNE2/',str(i),'.png']), dpi=200)
    plt.show()





##%%
#for i in range(len(N_test_choice)):
#    N_test = int(u_record_total[i][0])
#    u_pred_temp = u_record_total[i][1:]
#    plt.plot(X_test,u_pred_temp, linestyle = (0, (5, 5)), linewidth=1.5, color='black')
#    plt.savefig(''.join([PATH,str(case),str(test_Number),'_fcn',str(i),'.pdf']))    

    



