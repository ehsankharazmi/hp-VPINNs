"""
Deep VIV 

@Date: May 31 2019
@Author: Ehsan Kharazmi

"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec

np.random.seed(1234)
tf.set_random_seed(1234)

PATH = '/users/ekharazm/Documents/DeepVIV-NDP/Results/'

total_record = []
counter_glob = 0
act = 'sine'

###############################################################################
###############################################################################
class DeepVIV:

    def __init__(self, ZT_train, CF_train, layers):


        self.lb = ZT_train.min()
        self.ub = ZT_train.max()

        self.ZT_train = ZT_train
        self.CF_train = CF_train
        
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
#        self.sess = tf.Session()
#        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,\
                                                     log_device_placement=True,
                                                     device_count={'GPU': 0}))

        
        self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.z_tf  = tf.placeholder(tf.float64, shape=[None, 1])
        self.t_tf  = tf.placeholder(tf.float64, shape=[None, 1])
        self.cf_tf  = tf.placeholder(tf.float64, shape=[None, 1])

        
        # physics informed neural networks
        self.cf_pred  = self.net_cf(self.t_tf, self.z_tf)
       
        # loss
        self.lossv  = tf.reduce_mean(tf.square(self.cf_tf - self.cf_pred))
        self.loss   = self.lossv 


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        # optimizers
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

###############################################################################
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
#        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            if act == 'sine':
                H = tf.sin(tf.add(tf.matmul(H, W), b))
            if act == 'tanh':
                H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            

    def net_cf(self, t, x):
        cf    = self.neural_net(tf.concat([t,x],1), self.weights, self.biases)
        return cf
       
    def net_VIV(self, t, x):
        v    = self.neural_net(tf.concat([t,x],1), self.weights, self.biases)
        d1x_v  = tf.gradients(v, x)[0]
        d2x_v  = tf.gradients(d1x_v, x)[0]
        d3x_v  = tf.gradients(d2x_v, x)[0]
        d4x_v  = tf.gradients(d3x_v, x)[0]
        d1t_v  = tf.gradients(v, t)[0]
        d2t_v  = tf.gradients(d1t_v, t)[0]
        fl   = self.mass/(self.DT**2)*d2t_v + self.beta/self.DT*d1t_v \
             - 1000*self.Tension/(self.DZ**2)*d2x_v + 100*self.EI/(self.DZ**4)*d4x_v
        return fl


#################################################################################    
#    def train(self, num_epochs, batch_size, n_iter, learning_rate):
#
#        counter = 0
##        total_record = []
#
#        win_num = 0
#        ZT_train = self.ZT_train_window[win_num]
#        v_train  = self.v_train_window[win_num]
#        F_l_train= self.F_l_train_window[win_num]
#        
#        z   = ZT_train[:,0:1]
#        t   = ZT_train[:,1:2]
#        v   = v_train
#        fl  = F_l_train
#
#        for epoch in range(num_epochs):
#            
#            N = ZT_train.shape[0]
#            perm = np.random.permutation(N)
#            N_batches = int(np.floor(N/batch_size))
#            
#            start_time = time.time()
#            for batch in range(0, N_batches*batch_size, batch_size):
#                idx = perm[np.arange(batch, batch+batch_size)]
#                (z_batch, t_batch, v_batch) =  (z[idx,:], t[idx,:], v[idx,:])
#                (z_f_batch, t_f_batch, fl_f_batch) =  (z[idx,:], t[idx,:], fl[idx,:])
#
#                tf_dict = {self.z_tf: z_batch, self.t_tf: t_batch, self.v_tf: v_batch,\
#                           self.z_f_tf: z_f_batch, self.t_f_tf: t_f_batch, self.fl_f_tf: fl_f_batch,\
#                           self.learning_rate: learning_rate}
#                
#                for it in range(n_iter):
#                    self.sess.run(self.train_op, tf_dict)
#                    counter = counter + 1
#
#                    if it % 100 == 0:
#                        elapsed = time.time() - start_time
#                        beta_value = self.sess.run(self.beta, tf_dict)
##                        Tension_value = self.sess.run(self.Tension, tf_dict)
##                        EI_value = self.sess.run(self.EI, tf_dict)
##                        beta_value = self.beta
#                        Tension_value = self.Tension
#                        EI_value = self.EI
#                        loss_value, lossu_value, lossfl_value = self.sess.run([self.loss, self.lossv, self.lossfl], tf_dict)
#                        total_record.append(np.array([counter, loss_value, beta_value, Tension_value, EI_value]))
#                        print('Epoch: %d, batch: %d, It: %d, Lossv: %.5e, Lossfl: %.5e, beta: %.3f, EI: %.3f, Tension: %.3f, Time: %.2f'
#                              %(epoch, batch/batch_size, it, lossu_value, lossfl_value, beta_value, EI_value, Tension_value, elapsed, ))
#                        start_time = time.time()
#
#        return total_record


################################################################################    
    def train_adam(self, num_epochs, batch_size_cf, n_iter, learning_rate):

        counter = 0
#        total_record = []

        z   = self.ZT_train[:,0:1]
        t   = self.ZT_train[:,1:2]
        CF   = self.CF_train

        start_time = time.time()
        for epoch in range(num_epochs):
            
#            idx_cf = np.random.choice(ZT_train.shape[0], batch_size_cf, replace=False)
#            (z_batch, t_batch, CF_batch) =  (z[idx_cf,:], t[idx_cf,:], CF[idx_cf,:])
            (z_batch, t_batch, CF_batch) =  (z, t, CF)

            tf_dict = {self.z_tf: z_batch, self.t_tf: t_batch, self.cf_tf: CF_batch,\
                       self.learning_rate: learning_rate}            

            for it in range(n_iter):
                self.sess.run(self.train_op, tf_dict)
                counter = counter + 1

                if it % 100 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    total_record.append(np.array([counter, loss_value]))
                    print('Epoch: %d, It: %d, Loss: %.4e, Time: %.4f'
                          %(epoch, it, loss_value, elapsed))
                    start_time = time.time()

        
        
        return total_record

################################################################################    
    def train(self, num_epochs, batch_size_cf, n_iter, learning_rate):

        z   = self.ZT_train[:,0:1]
        t   = self.ZT_train[:,1:2]
        CF   = self.CF_train
       
        tf_dict = {self.z_tf: z, self.t_tf: t, self.cf_tf: CF,\
                   self.learning_rate: learning_rate}
        
        self.optimizer.minimize(self.sess, feed_dict = tf_dict,\
                                fetches = [self.loss],\
                                loss_callback = self.callback)
        
        
    def callback(self, loss):
        total_record.append(np.array([loss]))
        print('Loss: %.5e'%(loss))    
###############################################################################    
    def predict(self, ZT_star):
        
        tf_dict = {self.z_tf: ZT_star[:,0:1], self.t_tf: ZT_star[:,1:2]}
        cf_pred  = self.sess.run(self.cf_pred, tf_dict)
               
        return cf_pred
    
################################################################################
   
# =============================================================================
#                               Main
# =============================================================================    
if __name__ == "__main__": 
    
    layers  = [2] + 4*[50] + [1]
    
    ################################################################################
    data_total = scipy.io.loadmat('./NDP/Data - Uniform - Bare/test2030.mat')
    freq_measurment = int(data_total['fs'].flatten()[0])
    chan_names = data_total['chan_names']
    ACC_total = np.transpose(data_total['data'][78:86,:])
#    CF_total = data_total['data']
    CF_total = np.transpose(data_total['data'][46:70,:])

    z_sensors = np.array([2.555,3.084,3.224,4.155,6.030,8.609,8.889,10.285,13.676,\
                          16.452,16.891,19.997,20.193,21.393,22.460,23.165,25.153,\
                          26.254,28.863,29.365,31.191,33.005,36.559,37.322])
    z_star = z_sensors
    t_star = np.arange(0, CF_total.shape[0]/freq_measurment, 1/freq_measurment)
    
#    z_sensors = np.arange(-1, 1, 10/10)
#    t_star = np.arange(0, 10, 1/100)
    Z, T = np.meshgrid(z_sensors, t_star)
    ZT_star = np.hstack((Z.flatten()[:,None], T.flatten()[:,None]))
    CF_star = CF_total.flatten()[:,None]

    ## Constructing the training data sets
    Ini_z_idx = 0
    Fin_z_idx = 23
    Ini_z = z_star[Ini_z_idx]
    Fin_z = z_star[Fin_z_idx]

    Ini_time_idx = 50000
    Fin_time_idx = 51000
    time_sampling = 1
    Ini_time = t_star[Ini_time_idx]
    Fin_time = t_star[Fin_time_idx]

    T_window = T[Ini_time_idx : Fin_time_idx : time_sampling, Ini_z_idx : Fin_z_idx + 1]
    Z_window = Z[Ini_time_idx : Fin_time_idx : time_sampling, Ini_z_idx : Fin_z_idx + 1]
    CF_window = CF_total[Ini_time_idx : Fin_time_idx : time_sampling, Ini_z_idx : Fin_z_idx + 1]
    ZT_star_window  = np.hstack((Z_window.flatten()[:,None], T_window.flatten()[:,None]))
    CF_star_window = CF_window.flatten()[:,None]
    ################################################################################
#    N_train = 1000 * 24 
    N_train = len(ZT_star_window)
    idx = np.random.choice(ZT_star_window.shape[0], N_train, replace=False)
    ZT_train = ZT_star_window[idx,:]
    CF_train  = CF_star_window[idx,:]#/max(abs(CF_star_window))

    model = DeepVIV(ZT_train, CF_train, layers)


#%%    

    batch_size_cf = int(100/100*N_train)

    test_Number = 'set-shortwindow-LR03_Actsine'
    test_Number =''.join([test_Number, '_L',str(len(layers)-2),'N',str(layers[1]),\
                          '_NT',str(len(ZT_train)),'_BSv',str(batch_size_cf), '_'])

#%%
    total_record_1 = model.train(num_epochs = 50000, batch_size_cf = batch_size_cf, n_iter = 0 + 1, learning_rate = 1e-3)
#    total_record_1 = model.train(num_epochs = 50000, batch_size_cf = batch_size_cf, n_iter = 0 + 1, learning_rate = 1e-3)
#    total_record_1 = model.train(num_epochs = 50000, batch_size_cf = batch_size_cf, n_iter = 0 + 1, learning_rate = 1e-3)
#    total_record_1 = model.train(num_epochs = 50000, batch_size_cf = batch_size_cf, n_iter = 0 + 1, learning_rate = 1e-3)
#    total_record_1 = model.train(num_epochs = 50000, batch_size_cf = batch_size_cf, n_iter = 0 + 1, learning_rate = 1e-3)

#%%
    CF_pred = model.predict(ZT_star_window)
    
    t_test = t_star[Ini_time_idx: 60000]
    z_test = z_sensors #[Ini_z_idx : Fin_z_idx + 1]
    Z_test, T_test = np.meshgrid(z_test, t_test)
    ZT_test = np.hstack((Z_test.flatten()[:,None], T_test.flatten()[:,None]))
    CF_test = CF_total[Ini_time_idx: 60000,:].flatten()[:,None]    
    CF_pred_test = model.predict(ZT_test)
    
#%%    
    r = 0.0135
    N = 300
    DZ = (z_sensors.max() - z_sensors.min())/(N-1)
    z_int = np.linspace(z_sensors.min(), z_sensors.max(), N)
    t_int = t_star[Ini_time_idx:Fin_time_idx]
    Z_int, T_int = np.meshgrid(z_int, t_int)
    ZT_int = np.hstack((Z_int.flatten()[:,None], T_int.flatten()[:,None]))
    CF_pred_int = model.predict(ZT_int)
    CF_pred_int = np.asarray(np.split(CF_pred_int.flatten(), len(t_int)))

    z_tf  = tf.placeholder(tf.float64, shape=[None, 1])
    t_tf  = tf.placeholder(tf.float64, shape=[None, 1])
    epsilon    = model.net_cf(t_tf, z_tf)
    epsilon_t  = tf.gradients(epsilon, t_tf)[0]
    epsilon_tt = tf.gradients(epsilon_t, t_tf)[0]
    epsilon_pred    = model.sess.run(epsilon   , {t_tf: ZT_int[:,1][:,None], z_tf : ZT_int[:,0][:,None]})
    epsilon_t_pred  = model.sess.run(epsilon_t , {t_tf: ZT_int[:,1][:,None], z_tf : ZT_int[:,0][:,None]})
    epsilon_tt_pred = model.sess.run(epsilon_tt, {t_tf: ZT_int[:,1][:,None], z_tf : ZT_int[:,0][:,None]})

    epsilon_tt_pred = np.asarray(np.split(epsilon_tt_pred.flatten(), len(t_int)))


#%%
    A = np.zeros([N,N])
    for i in range(0,N):
        if i == 0:
            A[0,0:2] = [-2,1]
        elif i == N-1:
            A[N-1,-2:] = [1,-2]
        else:
            A[i,i-1:i+2] = [1,-2,1]
    A = A*r/(DZ**2)
#%%
    y = []
    for i in range(len(t_int)):
        b = CF_pred_int[i,:]
        y_time = np.linalg.solve(A,b)
        y.append(y_time)

    y2 = np.array(y)

#%%
    N = y2.shape[0]
    Dt = 1/(freq_measurment)
    AA = np.zeros([N,N])
    for i in range(0,N):
        if i == 0:
            AA[0,0:2] = [-2,1]
        elif i == N-1:
            AA[N-1,-2:] = [1,-2]
        else:
            AA[i,i-1:i+2] = [1,-2,1]
    AA = AA/(Dt**2)
    
    acc_from_y = np.matmul(AA, y2)

    y_from_acc = []
    for i in range(8):
        b = ACC_total[Ini_time_idx: Fin_time_idx,i]
        y_temp = np.linalg.solve(AA,b)
        y_from_acc.append(y_temp)

    y_from_acc = np.transpose(np.array(y_from_acc))


#%%
    acc = []
    for i in range(len(t_int)):
        b = epsilon_tt_pred[i,:]/(freq_measurment/10)**2
        acc_time = np.linalg.solve(A,b)
        acc.append(acc_time)
    
    acc2 = np.array(acc)

#%%
    with open(''.join([PATH,str(test_Number),'disacc_history','.mat']), 'wb') as f:
        scipy.io.savemat(f, {'dis'    : y2})
        scipy.io.savemat(f, {'acc'    : acc2})



#%%
    ACC_test = ACC_total[Ini_time_idx: Fin_time_idx,:]
    
    tstep_test = np.arange(Ini_time_idx, Fin_time_idx, 1)
    zlocs = [0,1,2,3,4,5,6,7]
    intlocs = [14,52,96,123,162,194,226,262]
    for i in range(4,5):
        zloc = zlocs[i]
        intloc = intlocs[i]
        fig, ax = plt.subplots()
        plt.plot(tstep_test, ACC_test[:,zloc], 'b', label= 'measurment')
#        plt.plot(tstep_test, 1.44e-4*acc2[:,intloc], 'k:', label = 'prediction')
        plt.plot(tstep_test[1:-1], 1e-6*acc_from_y[1:-1,intloc], 'r--', label = 'prediction')
        ax.tick_params(axis="x", labelsize = 14)
        ax.tick_params(axis="y", labelsize = 14)
#        ax.axvline(55000, linewidth=3, linestyle='--', color='k')
        plt.grid(True)
#        ax.set_xlabel('$time \, step$', fontsize = 20)
#        ax.set_ylabel('$acceleration$', fontsize = 20)
#        ax.set_title(''.join(['$z = %.2f$']) % (z_sensors[zloc]), fontsize = 20)
#        ax.set_xlim(41, 50)
#        ax.set_ylim(-100, 100)
        ax.locator_params(nbins=6)
        legend = plt.legend(shadow=True, loc='upper right', fontsize=12)
        fig.set_size_inches(w=10,h=2.2)
        fig.tight_layout()
#        plt.savefig(''.join([PATH,'FullyCon_Single_Sensor_Training/',str(test_Number),'ACCchannel',str(zloc+1),'.png']),dpi = 300)




#%%
    tstep_test = np.arange(Ini_time_idx, Fin_time_idx, 1)
    zlocs = [0,1,2,3,4,5,6,7]
    intlocs = [14,52,96,123,162,194,226,262]
    for i in range(4,5):
        zloc = zlocs[i]
        intloc = intlocs[i]
        fig, ax = plt.subplots()
    #    plt.plot(tstep_test, y2[:,intloc]*1e-6/(2*r*r**2), 'r--', label = 'prediction')
        plt.plot(tstep_test, y2[:,intloc]*1e-6/(2*r), 'r--', label = 'prediction')
        plt.plot(tstep_test, y_from_acc[:,zloc]/(2*r), 'k:', label = 'prediction')
    #    plt.plot(t_star[Ini_time_idx:Fin_time_idx], CF_total[Ini_time_idx:Fin_time_idx,zloc], 'b', label= 'measurment')
        ax.tick_params(axis="x", labelsize = 14)
        ax.tick_params(axis="y", labelsize = 14)
    #    ax.axvline(55000, linewidth=3, linestyle='--', color='k')
        plt.grid(True)
    #    ax.set_xlabel('$time \, step$', fontsize = 20)
    #    ax.set_ylabel('$acceleration$', fontsize = 20)
    #    ax.set_title(''.join(['$z = %.2f$']) % (z_sensors[zloc]), fontsize = 20)
    #    ax.set_xlim(41, 50)
    #    ax.set_ylim(-100, 100)
        ax.locator_params(nbins=6)
        legend = plt.legend(shadow=True, loc='upper right', fontsize=12)
        fig.set_size_inches(w=10,h=2.2)
        fig.tight_layout()
    #    plt.savefig(''.join([PATH,'FullyCon_Single_Sensor_Training/',str(test_Number),'ACCchannel',str(zloc+1),'.png']),dpi = 300)



#%%

################################################################################
#       plotting
################################################################################

    font = 24
    dpi = 400

    CF_pred_test_plot = np.asarray(np.split(CF_pred_test.flatten(), len(t_test)))
    CF_test_plot = np.asarray(np.split(CF_test.flatten(), len(t_test)))

#%%
    tstep_test = np.arange(50000, 60000, 1)
#    for zloc in range(len(z_test)):
    for zloc in [12]:
        fig, ax = plt.subplots()
        plt.plot(tstep_test, CF_test_plot[:,zloc], 'b', label= 'measurment')
        plt.plot(tstep_test, CF_pred_test_plot[:,zloc], 'r--', label = 'prediction')
        ax.tick_params(axis="x", labelsize = 14)
        ax.tick_params(axis="y", labelsize = 14)
        ax.axvline(55000, linewidth=3, linestyle='--', color='k')
        plt.grid(True)
#        ax.set_xlabel('$t$', fontsize = 20)
#        ax.set_ylabel('$y$', fontsize = 20)
#        ax.set_title(''.join(['$z = %.2f$']) % (z_sensors[zloc]), fontsize = 20)
#        ax.set_xlim(41, 50)
        ax.set_ylim(-100, 100)
        ax.locator_params(nbins=6)
        legend = plt.legend(shadow=True, loc='upper right', fontsize=12)
        fig.set_size_inches(w=25,h=4)
#        fig.tight_layout()
#        plt.savefig(''.join([PATH,'FullyCon_Single_Sensor_Training/',str(test_Number),'extrapolate_CFchannel',str(zloc+1),'.pdf']))



#%%
    CF_pred_plot = np.asarray(np.split(CF_pred.flatten(), (Fin_time_idx-Ini_time_idx)/time_sampling))
    for zloc in range(len(z_test)):
#    for zloc in [12]:
        fig, ax = plt.subplots()
        plt.plot(t_star[Ini_time_idx:Fin_time_idx], CF_total[Ini_time_idx:Fin_time_idx,zloc], 'b', label= 'measurment')
        plt.plot(t_star[Ini_time_idx:Fin_time_idx:time_sampling], CF_pred_plot[:,zloc], 'r--', label = 'prediction')
        ax.tick_params(axis="x", labelsize = 18)
        ax.tick_params(axis="y", labelsize = 18)
        ax.set_xlabel('$t$', fontsize = 20)
        ax.set_ylabel('$y$', fontsize = 20)
        ax.set_title(''.join(['$z = %.2f$']) % (z_sensors[zloc]), fontsize = 20)
    #    ax.set_ylim(-2, 2)
        ax.locator_params(nbins=5)
        legend = plt.legend(shadow=True, loc='lower right', fontsize=13)
        fig.set_size_inches(w=12,h=5)
        fig.tight_layout()   
        plt.savefig(''.join([PATH,str(test_Number),'pred_senloc',str(zloc+1),'.pdf']))

#%%
    for zloc in range(len(z_test)):
        print(max(abs(CF_total[Ini_time_idx:Fin_time_idx:10,zloc] - CF_pred_plot[:,zloc]))/max(abs(CF_total[Ini_time_idx:Fin_time_idx:10,zloc])))
 

#%%
    
    ################################################################################
    #       Loss and identified parameters
    ################################################################################

#    total_record_plot = np.asarray(total_record)
#
##    total_record_plot = np.concatenate([total_record_1, total_record_2, total_record_3, total_record_4])
#    iteration = np.arange(0, 1*total_record_plot.shape[0], 1)
##    iteration   = total_record_plot[:,0]
#    loss_his    = total_record_plot[:,1]
#    
#    font = 16
#    fig, ax = plt.subplots()
#    plt.plot(iteration, loss_his)
#    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
#    plt.xlabel('$iteration$' , fontsize = font)
#    plt.ylabel('$loss$' , fontsize = font)
#    plt.yscale('log')
#    plt.grid(True)
##    plt.savefig(''.join([PATH,str(test_Number),'loss','.pdf']))

#%%

    font = 16
    fig, ax = plt.subplots()
    plt.plot(total_record)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$' , fontsize = font)
    plt.ylabel('$loss$' , fontsize = font)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(''.join([PATH,str(test_Number),'loss','.pdf']))

#%%
    trained_weights = model.sess.run(model.weights)
    trained_biases = model.sess.run(model.biases)
    trained_biases_2 = []
    for i in range(len(trained_biases)):
        trained_biases_2.append(trained_biases[i].flatten())

    with open(''.join([PATH,str(test_Number),'history','.mat']), 'wb') as f:
        scipy.io.savemat(f, {'loss'    : total_record})
        scipy.io.savemat(f, {'t_test'    : t_test})
        scipy.io.savemat(f, {'t_star'    : t_star})
        scipy.io.savemat(f, {'CF_pred_test_plot' : CF_pred_test_plot})
        scipy.io.savemat(f, {'CF_pred_plot'      : CF_pred_plot})
        scipy.io.savemat(f, {'CF_test_plot'      : CF_test_plot})
        scipy.io.savemat(f, {'CF_total' : CF_total})
        scipy.io.savemat(f, {'weights'  : trained_weights})
        scipy.io.savemat(f, {'biases'   : trained_biases_2})
        scipy.io.savemat(f, {'activation'  : act})
        scipy.io.savemat(f, {'layers'  : layers})



#%%



















#%%
#    z_star_plot = np.asarray(np.split(ZT_star[:,0:1].flatten(),len(t_star)))
#    t_star_plot = np.asarray(np.split(ZT_star[:,1:2].flatten(),len(t_star)))
#    v_star_plot = np.asarray(np.split(CF_star.flatten(),len(t_star)))
#
#    font = 22
#    fig, ax = plt.subplots()
#    levels = 20
#    CS_test = ax.contourf(t_star_plot, z_star_plot, v_star_plot, levels=levels, cmap='jet', origin='lower')
#    cbar = fig.colorbar(CS_test)
#    ax.set_xlabel('$t$', fontsize = font)
#    ax.set_ylabel('$z$', fontsize = font)
#    ax.set_title('$ CF $', fontsize = font)
#    for s in z_sensors:
#        ax.axhline(s, linewidth=1 , linestyle='-', color='k')
##    ax.set_xlim(378, 537)
##    ax.set_ylim(0, 5)
#    ax.tick_params(axis="x", labelsize = 18)
#    ax.tick_params(axis="y", labelsize = 18)
#    lims_x = ax.get_xlim()
#    lims_y = ax.get_ylim()
#    fig.set_size_inches(w=12, h=8)
    
    
