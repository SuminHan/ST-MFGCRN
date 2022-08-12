import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules import *
from dcgru_cell_tf2 import *
from DeepSTN_net import *
from STResNet import *

def row_normalize(an_array):
    sum_of_rows = an_array.sum(axis=1)
    normalized_array = an_array / sum_of_rows[:, np.newaxis]
    return normalized_array


class MyDNNLayer(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDNNLayer, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XCT = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XPT = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQT = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        

    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        XC = self.FC_XC0(XC)
        XC = tf.transpose(XC, perm=[0, 3, 2, 1])
        XC = self.FC_XCT(XC)
        XC = tf.transpose(XC, perm=[0, 3, 2, 1])
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP)
        XP = tf.transpose(XP, perm=[0, 3, 2, 1])
        XP = self.FC_XPT(XP)
        XP = tf.transpose(XP, perm=[0, 3, 2, 1])
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ)
        XQ = tf.transpose(XQ, perm=[0, 3, 2, 1])
        XQ = self.FC_XQT(XQ)
        XQ = tf.transpose(XQ, perm=[0, 3, 2, 1])
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y
    

class MyDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.optcpt = extdata['optcpt']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
    
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])

    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        dayofweek = tf.one_hot(TEC[..., 0], depth = 7)
        timeofday = tf.one_hot(TEC[..., 1], depth = 24)
        minuteofday = tf.one_hot(TEC[..., 2], depth = 4)
        holiday = tf.one_hot(TEC[..., 3], depth = 1)
        TEC = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TEC = tf.expand_dims(TEC, axis = 2)

        # TEC = tf.expand_dims(TEC, 2)
        TEC = tf.tile(TEC, (1, 1, self.num_nodes, 1))
        XC = tf.concat((XC, TEC), -1)

        XC = self.FC_XC0(XC)
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        Y = XC
        
        # XP = self.FC_XP0(XP)
        # XP = self.FC_XP_DCGRU(XP)
        # XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        # Y = Y + XP
    
        # XQ = self.FC_XQ0(XQ)
        # XQ = self.FC_XQ_DCGRU(XQ)
        # XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        # Y = Y + XQ
        
        return Y
    
class MyDCGRU_CPT(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_CPT, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.optcpt = extdata['optcpt']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
    
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])

    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        dayofweek = tf.one_hot(TEC[..., 0], depth = 7)
        timeofday = tf.one_hot(TEC[..., 1], depth = 24)
        minuteofday = tf.one_hot(TEC[..., 2], depth = 4)
        holiday = tf.one_hot(TEC[..., 3], depth = 1)
        TEC = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TEC = tf.expand_dims(TEC, axis = 2)

        # TEC = tf.expand_dims(TEC, 2)
        TEC = tf.tile(TEC, (1, 1, self.num_nodes, 1))
        XC = tf.concat((XC, TEC), -1)

        XC = self.FC_XC0(XC)
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        Y = XC
        
        XP = self.FC_XP0(XP)
        XP = self.FC_XP_DCGRU(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        Y = Y + XP
    
        XQ = self.FC_XQ0(XQ)
        XQ = self.FC_XQ_DCGRU(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        Y = Y + XQ
        
        return Y
    

class MyDCGRU_STE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_STE, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        
        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
        
        XC = self.FC_XC0(XC) + STEC
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        # XP = self.FC_XP0(XP) + STEP
        # XP = self.FC_XP_DCGRU(XP)
        # XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        # XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = self.FC_XQ_DCGRU(XQ)
        # XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        Y = XC #+ XP + XQ
        return Y
    
class MyDCGRU_STE_CPT(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_STE_CPT, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        
        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
        
        XC = self.FC_XC0(XC) + STEC
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        Y = XC
        
        XP = self.FC_XP0(XP) + STEP
        XP = self.FC_XP_DCGRU(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        Y = Y + XP
        
        XQ = self.FC_XQ0(XQ) + STEQ
        XQ = self.FC_XQ_DCGRU(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        Y = Y + XQ
        
        return Y
    
class MyDCGRU_GST(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_GST, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])

    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        XC = self.FC_XC0(XC)
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP)
        XP = self.FC_XP_DCGRU(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ)
        XQ = self.FC_XQ_DCGRU(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y
    


    
# class MyDCGRU_GST_TE(tf.keras.layers.Layer):
#     def __init__(self, extdata, args):
#         super(MyDCGRU_GST_TE, self).__init__()
#         self.ext_bus = extdata['BUS_INFO']
#         self.ext_ent = extdata['ENT_EMP']
#         self.ext_pop = extdata['POP']
#         self.ext_adjpr = row_normalize(extdata['adj_pr'])
#         self.num_nodes = extdata['num_nodes']
#         self.D = args.D
        
#     def build(self, input_shape):
#         D = self.D
#         self.FC_XC0 = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(D)])
#         self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
#         self.FC_XC = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(2)])
        
#         self.FC_XP0 = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(D)])
#         self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
#         self.FC_XP = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(2)])
        
#         self.FC_XQ0 = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(D)])
#         self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
#         self.FC_XQ = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(2)])
        
#     def call(self, XC, TEC, WC, XP, TEP, WP, XQ, TEQ, WQ, TEY):
#         TEC = tf.expand_dims(TEC, axis=2)
#         TEC = tf.tile(TEC, multiples=[1, 1, self.num_nodes, 1])
#         TEP = tf.expand_dims(TEP, axis=2)
#         TEP = tf.tile(TEP, multiples=[1, 1, self.num_nodes, 1])
#         TEQ = tf.expand_dims(TEQ, axis=2)
#         TEQ = tf.tile(TEQ, multiples=[1, 1, self.num_nodes, 1])
        
#         print(XC.shape, TEC.shape)
        
#         XC = tf.concat((XC, TEC), -1)
#         XP = tf.concat((XP, TEP), -1)
#         XQ = tf.concat((XQ, TEQ), -1)
        
#         XC = self.FC_XC0(XC)
#         XC = self.FC_XC_DCGRU(XC)
#         XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
#         XC = self.FC_XC(XC)
        
#         XP = self.FC_XP0(XP)
#         XP = self.FC_XP_DCGRU(XP)
#         XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
#         XP = self.FC_XP(XP)
        
#         XQ = self.FC_XQ0(XQ)
#         XQ = self.FC_XQ_DCGRU(XQ)
#         XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
#         XQ = self.FC_XQ(XQ)
        
#         Y = XC + XP + XQ
#         return Y
    
    
    
class MyDCGRU_GST_STE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_GST_STE, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(units=D,K_diffusion=3,num_nodes=self.num_nodes), return_state=False)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        
        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
        
        #XC = tf.concat((XC, TEC), -1)
        #XP = tf.concat((XP, TEP), -1)
        #XQ = tf.concat((XQ, TEQ), -1)
        
        XC = self.FC_XC0(XC) + STEC
        XC = self.FC_XC_DCGRU(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        XP = self.FC_XP_DCGRU(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        XQ = self.FC_XQ_DCGRU(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y
    
    



class MyDCGRU_STE_ext(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyDCGRU_STE_ext, self).__init__()
        self.ext_bus = tf.cast(extdata['BUS_INFO'] / (extdata['BUS_INFO'].max(0)+1e-10), tf.float32)
        self.ext_ent = tf.cast(extdata['ENT_EMP'] / (extdata['ENT_EMP'].max(0)+1e-10), tf.float32)
        self.ext_pop = tf.cast(extdata['POP'] / (extdata['POP'].max(0)+1e-10), tf.float32)
        self.ext_lu = tf.cast(extdata['LU_TY'] / (extdata['LU_TY'].max(0)+1e-10), tf.float32)
        self.ext_local = tf.cast(extdata['LOCAL'] / (extdata['LOCAL'].max(0)+1e-10), tf.float32)
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.ext_adjpr1 = row_normalize(extdata['adj_pr1'])
        self.ext_adjpr2 = row_normalize(extdata['adj_pr2'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=[self.ext_adjpr], ext_feats=(self.ext_local, self.ext_pop, self.ext_lu), num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XP_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=[self.ext_adjpr], ext_feats=(self.ext_local, self.ext_pop, self.ext_lu),num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XQ_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=[self.ext_adjpr], ext_feats=(self.ext_local, self.ext_pop, self.ext_lu),num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        # XP = self.FC_XP0(XP) + STEP
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        # XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        # XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        # XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        Y = XC #+ XP + XQ
        return Y


def gaussian_noise_layer(input_layer, std=1):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise


class MyOurs_X_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOurs_X_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D

        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        ) 
        # self.FC_SE = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([ 
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        # SE = self.FC_SE(tf.concat((self.SE, self.bus_info), -1))
        # SE = self.FC_SE(self.loc)
 
        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU2, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y


class MyOurs_C_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOurs_C_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU2 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        # # XC1 = tf.reshape(XC1, (-1, TEC.shape[1], self.num_nodes, self.D))
        # # XC = XC + XC1
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU2, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        Y = XC 
        return Y


class MyOurs_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOurs_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y


class MyOursXMA_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursXMA_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        # self.CPTF = CPTFusion(D, self.out_dim)
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        # Y = self.CPTF(XC, XP, XQ)
        return Y


class MyOursOMA_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursOMA_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        # self.CPTF = CPTFusion(D, self.out_dim)
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        # Y = self.CPTF(XC, XP, XQ)
        return Y


class MyOursMAb(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursMAb, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        self.sentinel = args.sentinel # 'x', 'o'
        self.mgcn = args.mgcn # 'cat', 'mean'
        self.fusion = args.fusion # 'add', 'cat', 'weight'
        
        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = SE =self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])

        if self.sentinel == 'x' and self.mgcn == 'cat':
            self.FC_XC_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XP_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XQ_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        if self.sentinel == 'o' and self.mgcn == 'cat':
            self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        if self.sentinel == 'x' and self.mgcn == 'mean':
            self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        if self.sentinel == 'o' and self.mgcn == 'mean':
            self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
            self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        

        if self.fusion == 'add': 
            self.FC_XC = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.out_dim)])
            self.FC_XP = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.out_dim)])
            self.FC_XQ = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.out_dim)])
        if self.fusion == 'cat': 
            self.FC_Y = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.out_dim)])
        if self.fusion == 'weight': 
            self.CPTF = CPTFusion(D, self.out_dim)        
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        
        XP = self.FC_XP0(XP) + STEP
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        
        XQ = self.FC_XQ0(XQ) + STEQ
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))

        
        if self.fusion == 'add':
            XC = self.FC_XC(XC)
            XP = self.FC_XP(XP)
            XQ = self.FC_XQ(XQ)
            Y = XC + XP + XQ
        if self.fusion == 'cat':
            Y = self.FC_Y(tf.concat([XC, XP, XQ], -1))
        if self.fusion == 'weight':
            Y = self.CPTF(XC, XP, XQ)
        return Y

class MyOursGF2_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursGF2_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.CPTF = CPTFusion(D, self.out_dim)
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        # self.FC_XC = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell2(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        # Y = XC + XP + XQ
        Y = self.CPTF(XC, XP, XQ)
        return Y
        
class MyOursGF2X_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursGF2X_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.CPTF = CPTFusion(D, self.out_dim)
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        # self.FC_XC = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell2X(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        # Y = XC + XP + XQ
        Y = self.CPTF(XC, XP, XQ)
        return Y


class MyOursGF_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursGF_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.CPTF = CPTFusion(D, self.out_dim)
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        # self.FC_XC = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        # Y = XC + XP + XQ
        Y = self.CPTF(XC, XP, XQ)
        return Y


class MyOursXGF_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyOursXGF_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.CPTF = CPTFusion(D, self.out_dim)
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        # self.FC_XC = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        # Y = XC + XP + XQ
        Y = self.CPTF(XC, XP, XQ)
        return Y


class MyDCGRU_STEW_SEN_ext_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyDCGRU_STEW_SEN_ext_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STEW_layer = STEWmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU2 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XC_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU2 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU2 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ_DCGRU1 = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)

        STEC, STEP, STEQ = self.STEW_layer(self.SE, TEC, TEP, TEQ, WC, WP, WQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XC = gaussian_noise_layer(XC)
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False, return_sequences=True)(XC)
        XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU2, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False, return_sequences=True)(XP)
        XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU2, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False, return_sequences=True)(XQ)
        XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU2, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y


class MyDCGRU_STEDY_ext_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyDCGRU_STEDY_ext_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                adj = extdata[node]
                if extdata['zerodiag'] == 'True':
                    np.fill_diagonal(adj, 0)
                    
                self.ext_feats.append(tf.cast(adj / (adj.max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU1 = DCGRU_ADJ_DY_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats,num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XC_DCGRU2 = DCGRU_ADJ_DY_Cell(units=self.D,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                
        self.FC_XP_DCGRU1 = DCGRU_ADJ_DY_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP_DCGRU2 = DCGRU_ADJ_DY_Cell(units=self.D,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU1 = DCGRU_ADJ_DY_Cell(units=self.D,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ_DCGRU2 = DCGRU_ADJ_DY_Cell(units=self.D,adj_mats=self.adj_mats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        # self.SE = tf.zeros((self.num_nodes, self.D), dtype=tf.float32)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        XC = gaussian_noise_layer(XC)
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU1, return_state=False)((XC, XC))
        # XC = tf.reshape(XC, (-1, TEC.shape[1], self.num_nodes, self.D))
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU2, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        # XP = gaussian_noise_layer(XP)
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU1, return_state=False)((XP, XP))
        # XP = tf.reshape(XP, (-1, TEP.shape[1], self.num_nodes, self.D))
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU2, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = gaussian_noise_layer(XQ)
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU1, return_state=False)((XQ, XQ))
        # XQ = tf.reshape(XQ, (-1, TEQ.shape[1], self.num_nodes, self.D))
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU2, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y



class MyDCGRU_STEW_ext_ab(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyDCGRU_STEW_ext_ab, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max()+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STEW_layer = STEWmbedding(self.num_nodes, D)
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        WC = tf.cast(WC, tf.float32)
        WP = tf.cast(WP, tf.float32)
        WQ = tf.cast(WQ, tf.float32)


        STEC, STEP, STEQ = self.STEW_layer(self.SE, TEC, TEP, TEQ, WC, WP, WQ)
                
        XC = self.FC_XC0(XC) + STEC 
        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)
        
        XP = self.FC_XP0(XP) + STEP
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ0(XQ) + STEQ 
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        return Y


class MyDCGRU_STE_ext_sep(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_STE_ext_sep, self).__init__()
        self.extdata = extdata
        self.args = args
        
    def build(self, input_shape):
        self.mystgru1 = MyDCGRU_STE_ext(self.extdata, self.args, out_dim=1)
        self.mystgru2 = MyDCGRU_STE_ext(self.extdata, self.args, out_dim=1)
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        Y1 = self.mystgru1(XC[..., :1], TEC, XP[..., :1], TEP, XQ[..., :1], TEQ, TEY[..., :1])
        Y2 = self.mystgru1(XC[..., -1:], TEC, XP[..., -1:], TEP, XQ[..., -1:], TEQ, TEY[..., -1:])

        Y = tf.concat((Y1, Y2), -1)
        return Y



class Closeness_spatial(tf.keras.layers.Layer):
    def __init__(self, dist):
        super(Closeness_spatial, self).__init__()
        self.dist = tf.cast(dist / dist.max(), tf.float32)
        self.num_nodes = self.dist.shape[0]
        self.P = 3
        
    def build(self, input_shape):
        # t, n, _ = input_shape
        print('input_shape', input_shape)
        D = input_shape[-1]
        # self.mu = self.add_weight(
        #     shape=(self.P, ), initializer="random_normal", trainable=True, dtype=tf.float32, name='mu'
        # )
        self.FC_sigma = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1, activation="sigmoid")])
        self.FC_mu = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1, activation="sigmoid")])
        self.FC_alpha = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu")])
    
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        # XC: (b, t, n, D), STEC: (b, t, n, D)
        # adj_list = []
        shp = tf.shape(STEC)
        sigma = tf.tile(self.FC_sigma(STEC), [1,1,1,self.num_nodes])
        mu = tf.tile(self.FC_mu(STEC), [1,1,1,self.num_nodes])
        alpha = tf.tile(tf.nn.softmax(self.FC_alpha(STEC), 1), [1,1,1,shp[-1]])

        # print(sigma.shape, mu.shape, alpha.shape)

        dist = self.dist
        dist = tf.expand_dims(tf.expand_dims(dist, 0), 0)
        print(dist.shape)
        dist = tf.tile(dist, (shp[0], shp[1], 1, 1))
        print(dist.shape)



        
        SC = []
        for t in range(self.P):
            adj = tf.nn.softmax(((dist[:, t, ...] - mu[:, t, ...]) / sigma[:, t, ...])**2)
            # adj_list.append(adj)
            print('adj', adj.shape)
            adj_XC = adj@XC[:, t, :, :]
            # print('adj_XC', adj_XC.shape)
            SC.append(alpha[:, t, ...]*adj_XC)
            # SC.append(adj_XC)
        YC = self.FC_XC(tf.stack(SC, 1))
        print(YC.shape)
        return YC


class MyDCGRU_STE_lag(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyDCGRU_STE_lag, self).__init__()
        self.ext_bus = tf.cast(extdata['BUS_INFO'] / (extdata['BUS_INFO'].max(0)+1e-10), tf.float32)
        self.ext_ent = tf.cast(extdata['ENT_EMP'] / (extdata['ENT_EMP'].max(0)+1e-10), tf.float32)
        self.ext_pop = tf.cast(extdata['POP'] / (extdata['POP'].max(0)+1e-10), tf.float32)
        self.ext_lu = tf.cast(extdata['LU_TY'] / (extdata['LU_TY'].max(0)+1e-10), tf.float32)
        self.ext_local = tf.cast(extdata['LOCAL'] / (extdata['LOCAL'].max(0)+1e-10), tf.float32)
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.ext_adjpr1 = row_normalize(extdata['adj_pr1'])
        self.ext_adjpr2 = row_normalize(extdata['adj_pr2'])
        self.num_nodes = extdata['num_nodes']
        self.road_arr = row_normalize(extdata['ROAD_ARR'])
        self.dist = extdata['dist_arr']
        self.D = args.D
        self.d = args.d
        self.K = args.K
        self.L = args.L
        self.num_K = args.num_K
        self.out_dim = out_dim
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)  
        # self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer1 = TransformAttention(self.K, self.d)
        self.C_trans_layer2 = TransformAttention(self.K, self.d)
        # self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_spatial1 = Closeness_spatial(self.dist)
        self.C_spatial2 = Closeness_spatial(self.dist)

        self.FC_XC1 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC2 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
                        
        # self.FC_XC0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        self.FC_XC_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=[self.ext_adjpr], ext_feats=(self.ext_local, self.ext_pop, self.ext_lu), num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XCF = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])

        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XP_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=[self.ext_adjpr], ext_feats=(self.ext_local, self.ext_pop, self.ext_lu),num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XQ_DCGRU = DCGRU_ADJ_Cell(units=self.D,adj_mats=[self.ext_adjpr], ext_feats=(self.ext_local, self.ext_pop, self.ext_lu),num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)

        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        # STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC1 = self.FC_XC1(XC[..., :1]) #+ STEC
        XC2 = self.FC_XC2(XC[..., -1:]) #+ STEC
        XC1 = self.C_spatial1(XC1, STEC)
        XC2 = self.C_spatial2(XC2, STEC)
        # XC1 = self.C_trans_layer1(XC1, STEC, STEY)
        # XC2 = self.C_trans_layer2(XC2, STEC, STEY)
        XCS = tf.concat((XC1, XC2), -1)
        XCS = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XCS)
        XCS = tf.reshape(XCS, (-1, self.num_nodes, self.D))
        XCS = self.FC_XC(XCS)
        
        # XC = self.FC_XC0(XC) + STEC
        # XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        # XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XC = self.FC_XCF(XC)
        
        # XP = self.FC_XP0(XP) + STEP
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        # XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XP = self.FC_XP(XP)
        
        # XQ = self.FC_XQ0(XQ) + STEQ
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)
        # XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        # XQ = self.FC_XQ(XQ)
        
        # print(XCS.shape, XC.shape, XP.shape, XQ.shape)
        # Y = XCS + XC + XP + XQ
        Y = XCS
        return Y

def distant_matrix(n_neighbors, extdata):
    dist = extdata['dist_arr']
    n = dist.shape[0]

    e_in, e_out = [], []
    for i in range(n):
        e_in.append(np.argsort(dist[:, i])[:n_neighbors + 1])
        e_out.append(np.argsort(dist[i, :])[:n_neighbors + 1])
    e_in = np.array(e_in, dtype=np.int32)
    e_out = np.array(e_out, dtype=np.int32)

    adj_mat = np.array(dist)
    e_in = np.array(e_in)
    e_out = np.array(e_out)
    return adj_mat, e_in, e_out

def get_geo_feature(n_neighbors, extdata):
    # get locations
    loc = extdata['LOC']
    loc = (loc - np.mean(loc, axis=0)) / np.std(loc, axis=0)

    # get distance matrix
    dist, e_in, e_out = distant_matrix(n_neighbors, extdata)

    # normalize distance matrix
    n = loc.shape[0]
    edge = np.zeros((n, n))
    for i in range(n):
        for j in range(n_neighbors):
            edge[e_in[i][j], i] = edge[i, e_out[i][j]] = 1
    dist[edge == 0] = np.inf

    values = dist.flatten()
    values = values[values != np.inf]
    dist_mean = np.mean(values)
    dist_std = np.std(values)
    dist = np.exp(-(dist - dist_mean) / dist_std)

    # merge features
    features = []
    for i in range(n):
        f = np.concatenate([loc[i], dist[e_in[i], i], dist[i, e_out[i]]])
        features.append(f)
    features = np.stack(features)
    return features, (dist, e_in, e_out)


class MySTMetaNet(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MySTMetaNet, self).__init__()
        self.num_nodes = extdata['num_nodes']
        # self.ext_bus = extdata['BUS_INFO'] / (extdata['BUS_INFO'].max(0)+1e-10)
        # self.ext_ent = extdata['ENT_EMP'] / (extdata['ENT_EMP'].max(0)+1e-10)
        # self.ext_pop = extdata['POP'] / (extdata['POP'].max(0)+1e-10)
        # self.ext_local = extdata['LOCAL'] / (extdata['LOCAL'].max(0)+1e-10)
        # self.ext_lu = extdata['LU_TY'] / (extdata['LU_TY'].max(0)+1e-10)
        # self.ext_adjpr = row_normalize(extdata['adj_pr'])
        # self.ext_adjpr1 = row_normalize(extdata['adj_pr1'])
        # self.ext_adjpr2 = row_normalize(extdata['adj_pr2'])
        # self.dist_arr = extdata['dist_arr']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K

        self.n_neighbors = 7

        features, (dist, e_in, e_out) = get_geo_feature(n_neighbors=self.n_neighbors-1, extdata=extdata)

        print(features.shape, dist.shape, e_in.shape, e_out.shape)
        # self.features = features
        self.dist = dist
        self.e_in = e_in
        self.e_out = e_out
        
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max()+1e-10), tf.float32))

        self.node_feats = tf.concat(self.ext_feats, -1)

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(tf.cast(row_normalize(extdata[edge]), tf.float32))
        self.edge_feats = tf.stack(self.adj_mats, -1)
        
        e_in_feats = []
        e_out_feats = []
        for n in range(self.num_nodes):
            for j in self.e_in[n]:
                e_in_feats.append(self.edge_feats[n, j])
            for j in self.e_out[n]:
                e_out_feats.append(self.edge_feats[n, j])
        self.e_in_feats = tf.cast(e_in_feats, tf.float32)
        self.e_out_feats = tf.cast(e_out_feats, tf.float32)
        


        
        
    def build(self, input_shape):
        D = self.D
        # self.weight = self.add_weight(
        #     shape=(1, 1), initializer="random_normal", trainable=True, dtype=tf.float32, name='weight'
        # )
        self.SE = self.add_weight(
            shape=(self.num_nodes, D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        # self.W1 = self.add_weight(
        #     shape=(D, D), initializer="random_normal", trainable=True, dtype=tf.float32, name='W'
        # )
        # self.W2 = self.add_weight(
        #     shape=(D, D), initializer="random_normal", trainable=True, dtype=tf.float32, name='W'
        # )
        self.STE_layer = STEmbedding(self.num_nodes, D)

        
        self.FC_MKin = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        self.FC_MKout = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])


        # self.FC_NMK = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D),
        #                 layers.BatchNormalization()])
        # self.FC_EMK = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D),
        #                 layers.BatchNormalization()])
        # self.FC_SE = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.K, activation="softmax")])
        # self.FC_G_W = keras.Sequential([
        #                 layers.Dense(2*D*D),
        #                 layers.BatchNormalization()])
        # self.FC_G_b = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        # self.FC_SE_ent = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.K, activation="softmax")])
        # self.FC_SE_pop = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.K, activation="softmax")])
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        self.FC_XC_DCGRU1 = tf.keras.layers.RNN(
            DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(
            DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        # self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 1, self.num_nodes, 'laplacian'), return_state=False)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = tf.keras.layers.RNN(
            DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = tf.keras.layers.RNN(
            DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        
        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
        

        XC = self.FC_XC0(XC) + STEC


        # print('e_in', self.e_in.shape)
        onehot_in = tf.one_hot(self.e_in.reshape(-1), depth = self.num_nodes)
        onehot_out = tf.one_hot(self.e_out.reshape(-1), depth = self.num_nodes)

        # print('e_in_feats', self.e_in_feats.shape)

        Hi = tf.reshape(tf.tile(tf.expand_dims(XC, 3), [1, 1, 1, self.n_neighbors, 1]), (-1, 3, self.num_nodes*self.n_neighbors, self.D))
        Hj = onehot_in @ XC
        Hk = onehot_out @ XC

        # print(Hi.shape, Hj.shape)

        Hij = self.FC_MKin(tf.concat((Hi, Hj), -1))
        Hij = tf.nn.softmax(tf.reshape(Hij, (-1, 3, self.num_nodes, self.n_neighbors)), -1)
        
        Hik = self.FC_MKout(tf.concat((Hi, Hk), -1))
        Hik = tf.nn.softmax(tf.reshape(Hik, (-1, 3, self.num_nodes, self.n_neighbors)), -1)

        Hij = tf.expand_dims(Hij, -1)
        Hik = tf.expand_dims(Hik, -1)
        Hj = tf.reshape(Hj, (-1, 3, self.num_nodes, self.n_neighbors, self.D))
        Hk = tf.reshape(Hj, (-1, 3, self.num_nodes, self.n_neighbors, self.D))

        Hin = tf.reduce_sum(Hij * Hj, -2)
        Hout = tf.reduce_sum(Hik * Hk, -2)

        print(Hin.shape, Hout.shape)
        
        XC = self.FC_XC_DCGRU1(Hin) + self.FC_XC_DCGRU2(Hout)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)


        XP = self.FC_XP0(XP) + STEP
        XQ = self.FC_XQ0(XQ) + STEQ

        XP = self.FC_XP_DCGRU(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ_DCGRU(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ

        # Y = XC
        return Y



class MySTMetaNetTA(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MySTMetaNetTA, self).__init__()
        self.num_nodes = extdata['num_nodes']
        # self.ext_bus = extdata['BUS_INFO'] / (extdata['BUS_INFO'].max(0)+1e-10)
        # self.ext_ent = extdata['ENT_EMP'] / (extdata['ENT_EMP'].max(0)+1e-10)
        # self.ext_pop = extdata['POP'] / (extdata['POP'].max(0)+1e-10)
        # self.ext_local = extdata['LOCAL'] / (extdata['LOCAL'].max(0)+1e-10)
        # self.ext_lu = extdata['LU_TY'] / (extdata['LU_TY'].max(0)+1e-10)
        # self.ext_adjpr = row_normalize(extdata['adj_pr'])
        # self.ext_adjpr1 = row_normalize(extdata['adj_pr1'])
        # self.ext_adjpr2 = row_normalize(extdata['adj_pr2'])
        # self.dist_arr = extdata['dist_arr']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.num_K = args.num_K

        self.n_neighbors = 7

        features, (dist, e_in, e_out) = get_geo_feature(n_neighbors=self.n_neighbors-1, extdata=extdata)

        print(features.shape, dist.shape, e_in.shape, e_out.shape)
        # self.features = features
        self.dist = dist
        self.e_in = e_in
        self.e_out = e_out
        
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max()+1e-10), tf.float32))

        self.node_feats = tf.concat(self.ext_feats, -1)

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(tf.cast(row_normalize(extdata[edge]), tf.float32))
        self.edge_feats = tf.stack(self.adj_mats, -1)
        
        e_in_feats = []
        e_out_feats = []
        for n in range(self.num_nodes):
            for j in self.e_in[n]:
                e_in_feats.append(self.edge_feats[n, j])
            for j in self.e_out[n]:
                e_out_feats.append(self.edge_feats[n, j])
        self.e_in_feats = tf.cast(e_in_feats, tf.float32)
        self.e_out_feats = tf.cast(e_out_feats, tf.float32)
        


        
        
    def build(self, input_shape):
        D = self.D
        # self.weight = self.add_weight(
        #     shape=(1, 1), initializer="random_normal", trainable=True, dtype=tf.float32, name='weight'
        # )
        self.SE = self.add_weight(
            shape=(self.num_nodes, D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        # self.W = self.add_weight(
        #     shape=(D, D), initializer="random_normal", trainable=True, dtype=tf.float32, name='W'
        # )
        self.STE_layer = STEmbedding_Y(self.num_nodes, D)

        
        self.FC_MKin = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        self.FC_MKout = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])


        # self.FC_NMK = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D),
        #                 layers.BatchNormalization()])
        # self.FC_EMK = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D),
        #                 layers.BatchNormalization()])
        # self.FC_SE = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.K, activation="softmax")])
        # self.FC_G_W = keras.Sequential([
        #                 layers.Dense(2*D*D),
        #                 layers.BatchNormalization()])
        # self.FC_G_b = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        # self.FC_SE_ent = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.K, activation="softmax")])
        # self.FC_SE_pop = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.K, activation="softmax")])
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
                        
        # self.FC_XC_DCGRU1 = tf.keras.layers.RNN(
        #     DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        # self.FC_XC_DCGRU2 = tf.keras.layers.RNN(
        #     DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        
        self.C_trans_layer1 = TransformAttention(self.K, self.d)
        self.C_trans_layer2 = TransformAttention(self.K, self.d)

        # self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.ext_adjpr, 1, self.num_nodes, 'laplacian'), return_state=False)

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XP_DCGRU = tf.keras.layers.RNN(
            DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = tf.keras.layers.RNN(
            DCGRU_ADJ_Cell(units=D,adj_mats=self.adj_mats, ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K), return_state=False)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)

        print(TEC.shape, TEY.shape)
        
        
        STEC, STEP, STEQ, STEY = self.STE_layer(self.SE, TEC, TEP, TEQ, TEY)
        print(STEC.shape, STEP.shape, STEQ.shape)
        

        XC = self.FC_XC0(XC) + STEC


        # print('e_in', self.e_in.shape)
        onehot_in = tf.one_hot(self.e_in.reshape(-1), depth = self.num_nodes)
        onehot_out = tf.one_hot(self.e_out.reshape(-1), depth = self.num_nodes)

        # print('e_in_feats', self.e_in_feats.shape)

        Hi = tf.reshape(tf.tile(tf.expand_dims(XC, 3), [1, 1, 1, self.n_neighbors, 1]), (-1, 3, self.num_nodes*self.n_neighbors, self.D))
        Hj = onehot_in @ XC
        Hk = onehot_out @ XC

        # print(Hi.shape, Hj.shape)

        Hij = self.FC_MKin(tf.concat((Hi, Hj), -1))
        Hij = tf.nn.softmax(tf.reshape(Hij, (-1, 3, self.num_nodes, self.n_neighbors)), -1)
        
        Hik = self.FC_MKout(tf.concat((Hi, Hk), -1))
        Hik = tf.nn.softmax(tf.reshape(Hik, (-1, 3, self.num_nodes, self.n_neighbors)), -1)

        Hij = tf.expand_dims(Hij, -1)
        Hik = tf.expand_dims(Hik, -1)
        Hj = tf.reshape(Hj, (-1, 3, self.num_nodes, self.n_neighbors, self.D))
        Hk = tf.reshape(Hj, (-1, 3, self.num_nodes, self.n_neighbors, self.D))

        Hin = tf.reduce_sum(Hij * Hj, -2)
        Hout = tf.reduce_sum(Hik * Hk, -2)

        print(Hin.shape, Hout.shape)
        
        # XC = self.FC_XC_DCGRU1(Hin) + self.FC_XC_DCGRU2(Hout)
        
        XC = self.C_trans_layer1(Hin, STEC, STEY) + self.C_trans_layer2(Hout, STEC, STEY)
        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XC = self.FC_XC(XC)


        XP = self.FC_XP0(XP) + STEP
        XQ = self.FC_XQ0(XQ) + STEQ

        XP = self.FC_XP_DCGRU(XP)
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XP = self.FC_XP(XP)
        
        XQ = self.FC_XQ_DCGRU(XQ)
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ

        # Y = XC
        return Y





class MyGMAN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        
        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        
        
        XC = self.FC_XC0(XC) + STEC
        for i in range(self.L):
            XC = self.GSTAC_enc[i](XC, STEC)
        XC = self.C_trans_layer(XC, STEC, STEY)
        for i in range(self.L):
            XC = self.GSTAC_dec[i](XC, STEY)
        XC = self.FC_XC(XC)
        
        
        # XP = self.FC_XP0(XP) + STEP
        # for i in range(self.L):
        #     XP = self.GSTAP_enc[i](XP, STEP)
        # XP = self.P_trans_layer(XP, STEP, STEY)
        # for i in range(self.L):
        #     XP = self.GSTAP_dec[i](XP, STEY)
        # XP = self.FC_XP(XP)
        
        
        # XQ = self.FC_XQ0(XQ) + STEQ
        # for i in range(self.L):
        #     XQ = self.GSTAQ_enc[i](XQ, STEQ)
        # XQ = self.Q_trans_layer(XQ, STEQ, STEY)
        # for i in range(self.L):
        #     XQ = self.GSTAQ_dec[i](XQ, STEY)
        # XQ = self.FC_XQ(XQ)
        
        # Y = XC + XP + XQ
        Y = XC
        Y = tf.squeeze(Y, 1)
        return Y
    
class MyGMAN_CPT(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN_CPT, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        
        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        
        
        XC = self.FC_XC0(XC) + STEC
        for i in range(self.L):
            XC = self.GSTAC_enc[i](XC, STEC)
        XC = self.C_trans_layer(XC, STEC, STEY)
        for i in range(self.L):
            XC = self.GSTAC_dec[i](XC, STEY)
        XC = self.FC_XC(XC)
        
        
        XP = self.FC_XP0(XP) + STEP
        for i in range(self.L):
            XP = self.GSTAP_enc[i](XP, STEP)
        XP = self.P_trans_layer(XP, STEP, STEY)
        for i in range(self.L):
            XP = self.GSTAP_dec[i](XP, STEY)
        XP = self.FC_XP(XP)
        
        
        XQ = self.FC_XQ0(XQ) + STEQ
        for i in range(self.L):
            XQ = self.GSTAQ_enc[i](XQ, STEQ)
        XQ = self.Q_trans_layer(XQ, STEQ, STEY)
        for i in range(self.L):
            XQ = self.GSTAQ_dec[i](XQ, STEY)
        XQ = self.FC_XQ(XQ)
        
        Y = XC + XP + XQ
        # Y = XC
        Y = tf.squeeze(Y, 1)
        return Y


        

class MyGMAN_CPT_Fusion1(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN_CPT_Fusion1, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XC = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        self.CPTF = CPTFusion(D, 2)
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        
        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        
        
        XC = self.FC_XC0(XC) + STEC
        for i in range(self.L):
            XC = self.GSTAC_enc[i](XC, STEC)
        XC = self.C_trans_layer(XC, STEC, STEY)
        for i in range(self.L):
            XC = self.GSTAC_dec[i](XC, STEY)
        # XC = self.FC_XC(XC)
        
        
        XP = self.FC_XP0(XP) + STEP
        for i in range(self.L):
            XP = self.GSTAP_enc[i](XP, STEP)
        XP = self.P_trans_layer(XP, STEP, STEY)
        for i in range(self.L):
            XP = self.GSTAP_dec[i](XP, STEY)
        # XP = self.FC_XP(XP)
        
        
        XQ = self.FC_XQ0(XQ) + STEQ
        for i in range(self.L):
            XQ = self.GSTAQ_enc[i](XQ, STEQ)
        XQ = self.Q_trans_layer(XQ, STEQ, STEY)
        for i in range(self.L):
            XQ = self.GSTAQ_dec[i](XQ, STEY)
        # XQ = self.FC_XQ(XQ)
        
        XC = tf.squeeze(XC, 1)
        XP = tf.squeeze(XP, 1)
        XQ = tf.squeeze(XQ, 1)

        Y = self.CPTF(XC, XP, XQ)


        # Y = self.CPTF(tf.squeeze(tf.concat((XC, XP, XQ), -1), 1))
        # Y = XC + XP + XQ
        # Y = XC
        # Y = tf.squeeze(Y, 1)
        return Y


class MyGMAN_CPT_Fusion2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN_CPT_Fusion2, self).__init__()
        self.ext_bus = extdata['BUS_INFO']
        self.ext_ent = extdata['ENT_EMP']
        self.ext_pop = extdata['POP']
        self.ext_adjpr = row_normalize(extdata['adj_pr'])
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XC = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        self.CPTF = CPTFusion(D, 2)
        self.FC_Y = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']

        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        
        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        
        
        XC = self.FC_XC0(XC) + STEC
        for i in range(self.L):
            XC = self.GSTAC_enc[i](XC, STEC)
        XC = self.C_trans_layer(XC, STEC, STEY)
        for i in range(self.L):
            XC = self.GSTAC_dec[i](XC, STEY)
        # XC = self.FC_XC(XC)
        
        
        XP = self.FC_XP0(XP) + STEP
        for i in range(self.L):
            XP = self.GSTAP_enc[i](XP, STEP)
        XP = self.P_trans_layer(XP, STEP, STEY)
        for i in range(self.L):
            XP = self.GSTAP_dec[i](XP, STEY)
        # XP = self.FC_XP(XP)
        
        
        XQ = self.FC_XQ0(XQ) + STEQ
        for i in range(self.L):
            XQ = self.GSTAQ_enc[i](XQ, STEQ)
        XQ = self.Q_trans_layer(XQ, STEQ, STEY)
        for i in range(self.L):
            XQ = self.GSTAQ_dec[i](XQ, STEY)
        # XQ = self.FC_XQ(XQ)
        
        XC = tf.squeeze(XC, 1)
        XP = tf.squeeze(XP, 1)
        XQ = tf.squeeze(XQ, 1)

        Y = tf.concat((XC, XP, XQ), -1)
        Y = self.FC_Y(Y)
        # Y = self.CPTF(XC, XP, XQ)


        # Y = self.CPTF(tf.squeeze(tf.concat((XC, XP, XQ), -1), 1))
        # Y = XC + XP + XQ
        # Y = XC
        # Y = tf.squeeze(Y, 1)
        return Y

    
class MyGMAN_ext(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN_ext, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L

        self.adj_mats = []
        self.ext_feats = []
        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))
        
    def build(self, input_shape):
        D = self.D
        self.SE = self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer_Y = STEmbedding_Y(self.num_nodes, D)
        
        self.GSTAC_enc = [SGTAblock(self.K, self.d, self.adj_mats, self.ext_feats) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [SGTAblock(self.K, self.d, self.adj_mats, self.ext_feats) for _ in range(self.L)]
        
        # self.GSTAP_enc = [SGTAblock(self.K, self.d, self.adj_mats, self.ext_feats) for _ in range(self.L)]
        # self.P_trans_layer = TransformAttention(self.K, self.d)
        # self.GSTAP_dec = [SGTAblock(self.K, self.d, self.adj_mats, self.ext_feats) for _ in range(self.L)]
        
        # self.GSTAQ_enc = [SGTAblock(self.K, self.d, self.adj_mats, self.ext_feats) for _ in range(self.L)]
        # self.Q_trans_layer = TransformAttention(self.K, self.d)
        # self.GSTAQ_dec = [SGTAblock(self.K, self.d, self.adj_mats, self.ext_feats) for _ in range(self.L)]
        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D),
                        layers.BatchNormalization()])
        
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(2)])
        
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])
        
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(2)])

        # self.FC_WE = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        # print(self.ext_bus.shape, self.ext_ent.shape, self.ext_pop.shape)

        # STEC, STEP, STEQ, STEY = self.STE_layer_Y(SE, TEC, TEP, TEQ, TEY)
        STEC, STEP, STEQ, STEY = self.STE_layer_Y(self.SE, TEC, TEP, TEQ, TEY)
        
        XC = self.FC_XC0(XC)# + STEC
        for i in range(self.L):
            XC = self.GSTAC_enc[i](XC, STEC)
        XC = self.C_trans_layer(XC, STEC, STEY)
        for i in range(self.L):
            XC = self.GSTAC_dec[i](XC, STEY)
        XC = self.FC_XC(XC)
        
        
        # XP = self.FC_XP0(XP)# + STEP
        # for i in range(self.L):
        #     XP = self.GSTAP_enc[i](XP, STEP)
        # XP = self.P_trans_layer(XP, STEP, STEY)
        # for i in range(self.L):
        #     XP = self.GSTAP_dec[i](XP, STEY)
        # XP = self.FC_XP(XP)
        
        
        # XQ = self.FC_XQ0(XQ)# + STEQ
        # for i in range(self.L):
        #     XQ = self.GSTAQ_enc[i](XQ, STEQ)
        # XQ = self.Q_trans_layer(XQ, STEQ, STEY)
        # for i in range(self.L):
        #     XQ = self.GSTAQ_dec[i](XQ, STEY)
        # XQ = self.FC_XQ(XQ)
        
        # Y = XC
        Y = XC #+ XP + XQ
        Y = tf.squeeze(Y, 1)
        return Y
        

class MyConvRNN(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyConvRNN, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T

        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        self.Conv2D_C = Conv2DFilterPool(D)
        
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])  
        # self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        # self.Conv2D_P = Conv2DFilterPool(D)
        
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])
        # self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        # self.Conv2D_Q = Conv2DFilterPool(D)
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        # XP = self.FC_XP0(XP) + STEP
        # XQ = self.FC_XQ0(XQ) + STEQ

        XC = tf.reshape((self.converter_mat @ XC), (-1, self.H_height, self.H_width, self.D))
        # XP = tf.reshape((self.converter_mat @ XP), (-1, self.H_height, self.H_width, self.D))
        # XQ = tf.reshape((self.converter_mat @ XQ), (-1, self.H_height, self.H_width, self.D))

        XC = self.Conv2D_C(XC)
        # XP = self.Conv2D_P(XP)
        # XQ = self.Conv2D_Q(XQ)
        
        XC = tf.reshape(XC, (-1, C_num, self.H_height*self.H_width, self.D))
        # XP = tf.reshape(XP, (-1, P_num, self.H_height*self.H_width, self.D))
        # XQ = tf.reshape(XQ, (-1, Q_num, self.H_height*self.H_width, self.D))

        XC = self.converter_mat.T @ XC
        # XP = self.converter_mat.T @ XP
        # XQ = self.converter_mat.T @ XQ

        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        # XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        # XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)

        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))

        XC = self.FC_XC(XC)
        # XP = self.FC_XP(XP)
        # XQ = self.FC_XQ(XQ)

        Y = XC #+ XP + XQ
        return Y


class MyConvRNN2(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyConvRNN2, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T

        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XC_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XP0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])  
        self.FC_XP_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XP = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.FC_XQ0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FC_XQ_DCGRU = DCGRU_ADJ_SEN_Cell(units=self.D,SE=SE,adj_mats=self.adj_mats,ext_feats=self.ext_feats, num_nodes=self.num_nodes, num_K=self.num_K)
        self.FC_XQ = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        
        self.Conv2D_C = layers.Conv2D(D, kernel_size=(5,3), padding='same')#Conv2DFilterPool(D)
        self.Conv2D_P = layers.Conv2D(D, kernel_size=(5,3), padding='same')#Conv2DFilterPool(D)
        self.Conv2D_Q = layers.Conv2D(D, kernel_size=(5,3), padding='same')#Conv2DFilterPool(D)
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
                
        XC = self.FC_XC0(XC) + STEC
        XP = self.FC_XP0(XP) + STEP
        XQ = self.FC_XQ0(XQ) + STEQ

        XC = tf.reshape((self.converter_mat @ XC), (-1, self.H_height, self.H_width, self.D))
        XP = tf.reshape((self.converter_mat @ XP), (-1, self.H_height, self.H_width, self.D))
        XQ = tf.reshape((self.converter_mat @ XQ), (-1, self.H_height, self.H_width, self.D))

        XC = self.Conv2D_C(XC)
        XP = self.Conv2D_P(XP)
        XQ = self.Conv2D_Q(XQ)
        
        XC = tf.reshape(XC, (-1, C_num, self.H_height*self.H_width, self.D))
        XP = tf.reshape(XP, (-1, P_num, self.H_height*self.H_width, self.D))
        XQ = tf.reshape(XQ, (-1, Q_num, self.H_height*self.H_width, self.D))

        XC = self.converter_mat.T @ XC
        XP = self.converter_mat.T @ XP
        XQ = self.converter_mat.T @ XQ

        XC = tf.keras.layers.RNN(self.FC_XC_DCGRU, return_state=False)(XC)
        XP = tf.keras.layers.RNN(self.FC_XP_DCGRU, return_state=False)(XP)
        XQ = tf.keras.layers.RNN(self.FC_XQ_DCGRU, return_state=False)(XQ)

        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))

        XC = self.FC_XC(XC)
        XP = self.FC_XP(XP)
        XQ = self.FC_XQ(XQ)

        Y = XC + XP + XQ
        return Y

class MyConvLSTM(tf.keras.layers.Layer):
    def __init__(self, extdata, args, out_dim=2):
        super(MyConvLSTM, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T

        
    def build(self, input_shape):
        D = self.D
        self.SE = SE=self.add_weight(
            shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        )
        self.STE_layer = STEmbedding(self.num_nodes, D)        
        self.FC_XC0 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        # self.FC_XP0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])  
        # self.FC_XQ0 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])

        self.FC_XC = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.out_dim)])
        # self.FC_XP = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        # self.FC_XQ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.out_dim)])
        
        
        self.Conv2D_C = layers.ConvLSTM2D(
                filters=64,
                kernel_size=(5, 3),
                padding="same",
                return_sequences=False,
            )
        # self.Conv2D_P = layers.ConvLSTM2D(
        #         filters=64,
        #         kernel_size=(5, 3),
        #         padding="same",
        #         return_sequences=False,
        #         activation="relu",
        #     )
        # self.Conv2D_Q = layers.ConvLSTM2D(
        #         filters=64,
        #         kernel_size=(5, 3),
        #         padding="same",
        #         return_sequences=False,
        #         activation="relu",
        #     )
        
    def call(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        STEC, STEP, STEQ = self.STE_layer(self.SE, TEC, TEP, TEQ)
        print(STEC.shape, STEP.shape, STEQ.shape)
        
        XC = self.FC_XC0(XC) + STEC
        # XP = self.FC_XP0(XP) + STEP
        # XQ = self.FC_XQ0(XQ) + STEQ

        XC = tf.reshape((self.converter_mat @ XC), (-1, C_num, self.H_height, self.H_width, self.D))
        # XP = tf.reshape((self.converter_mat @ XP), (-1, P_num, self.H_height, self.H_width, self.D))
        # XQ = tf.reshape((self.converter_mat @ XQ), (-1, Q_num, self.H_height, self.H_width, self.D))

        XC = self.Conv2D_C(XC)
        # XP = self.Conv2D_P(XP)
        # XQ = self.Conv2D_Q(XQ)
        
        XC = tf.reshape(XC, (-1, self.H_height*self.H_width, self.D))
        # XP = tf.reshape(XP, (-1, self.H_height*self.H_width, self.D))
        # XQ = tf.reshape(XQ, (-1, self.H_height*self.H_width, self.D))

        XC = self.converter_mat.T @ XC
        # XP = self.converter_mat.T @ XP
        # XQ = self.converter_mat.T @ XQ

        XC = tf.reshape(XC, (-1, self.num_nodes, self.D))
        # XP = tf.reshape(XP, (-1, self.num_nodes, self.D))
        # XQ = tf.reshape(XQ, (-1, self.num_nodes, self.D))

        XC = self.FC_XC(XC)
        # XP = self.FC_XP(XP)
        # XQ = self.FC_XQ(XQ)

        Y = XC #+ XP + XQ
        return Y


class MyDeepSTN():
    def __init__(self, extdata, args, out_dim=2):
        # super(MyDeepSTN, self).__init__()
        from tensorflow.keras import backend as K
        K.set_image_data_format('channels_first')
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        self.poi_local = extdata['LU_TY']

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T
        

    def __call__(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        TE = tf.cast(TEY, tf.int32)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.tile(TE, (1, 1, self.H_height, self.H_width))

        XC = tf.reshape((self.converter_mat @ XC), (-1, C_num, self.H_height, self.H_width, 2))
        XP = tf.reshape((self.converter_mat @ XP), (-1, P_num, self.H_height, self.H_width, 2))
        XQ = tf.reshape((self.converter_mat @ XQ), (-1, Q_num, self.H_height, self.H_width, 2))

        ####################### DSTN ##################
        H=29
        W=14
        channel=2 #H-map_height W-map_width channel-map_channel
        c=6
        p=7
        t=3 #c-closeness p-period t-trend
        pre_F=64
        conv_F=64
        R_N=2 #pre_F-prepare_conv_featrue conv_F-resnet_conv_featrue R_N-resnet_number
        is_plus=True             #use ResPlus or mornal convolution
        is_plus_efficient=False  #use the efficient version of ResPlus
        plus=8
        rate=2            #rate-pooling_rate
        is_pt=False               #use PoI and Time or not
        P_N=self.poi_local.shape[-1]
        T_F=28
        PT_F=6
        T=TE.shape[1] #P_N-poi_number T_F-time_feature PT_F-poi_time_feature T-T_times/day 
        drop=0
        is_summary=True #show detail
        # lr=0.0002
        kernel1=1 #kernel1 decides whether early-fusion uses conv_unit0 or conv_unit1, 1 recommended
        isPT_F=1 #isPT_F decides whether PT_model uses one more Conv after multiplying PoI and Time, 1 recommended

        
        all_channel = channel * (c+p+t)
                
        cut0 = int( 0 )
        cut1 = int( cut0 + channel*c )
        cut2 = int( cut1 + channel*p )
        cut3 = int( cut2 + channel*t )
        
        c_input = tf.reshape(tf.transpose(XC, (0, 1, 4, 2, 3)), (-1, C_num*2, self.H_height, self.H_width))
        p_input = tf.reshape(tf.transpose(XP, (0, 1, 4, 2, 3)), (-1, P_num*2, self.H_height, self.H_width))
        t_input = tf.reshape(tf.transpose(XQ, (0, 1, 4, 2, 3)), (-1, Q_num*2, self.H_height, self.H_width))
        
        from tensorflow.keras.layers import Conv2D
        K.set_image_data_format('channels_first')
        c_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(c_input)
        p_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(p_input)
        t_out1=Conv2D(filters=pre_F,kernel_size=(1,1),padding="same")(t_input)
                
        if is_pt:
            poi_in = tf.reshape(self.converter_mat @ self.poi_local, (self.H_height, self.H_width, -1))
            poi_in = tf.transpose(poi_in, (2, 0, 1))
            poi_in = tf.cast(poi_in, tf.float32)
            poi_in = tf.expand_dims(poi_in, 0)
            poi_in = tf.tile(poi_in, (tf.shape(XC)[0], 1, 1, 1))
            # T_times/day + 7days/week 
            # time_in=Input(shape=(T+7,H,W))
            time_in = TE

            PT_model=PT_trans('PT_trans',P_N,PT_F,T,T_F,H,W,isPT_F)
            
            poi_time=PT_model([poi_in,time_in])
    
            cpt_con1=Concatenate(axis=1)([c_out1,p_out1,t_out1,poi_time])
            if kernel1:
                cpt=conv_unit1(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F),conv_F,drop,H,W)(cpt_con1)
            else:
                cpt=conv_unit0(pre_F*3+PT_F*isPT_F+P_N*(not isPT_F),conv_F,drop,H,W)(cpt_con1)
        
        else:
            cpt_con1=Concatenate(axis=1)([c_out1,p_out1,t_out1])
            if kernel1:
                cpt=conv_unit1(pre_F*3,conv_F,drop,H,W)(cpt_con1)
            else:
                cpt=conv_unit0(pre_F*3,conv_F,drop,H,W)(cpt_con1)  


        
        if is_plus:
            if is_plus_efficient:
                for i in range(R_N):
                    cpt=Res_plus_E('Res_plus_'+str(i+1),conv_F,plus,rate,drop,H,W)(cpt)
            else:
                for i in range(R_N):
                    cpt=Res_plus('Res_plus_'+str(i+1),conv_F,plus,rate,drop,H,W)(cpt)

        else:  
            for i in range(R_N):
                cpt=Res_normal('Res_normal_'+str(i+1),conv_F,drop,H,W)(cpt)

        cpt_conv2=Activation('relu')(cpt)
        cpt_out2=cpt_conv2 # cpt_out2=BatchNormalization()(cpt_conv2)
        cpt_conv1=Dropout(drop)(cpt_out2)
        cpt_conv1=Conv2D(filters=channel,kernel_size=(1, 1),padding="same")(cpt_conv1)
        # cpt_out1=Activation('tanh')(cpt_conv1)
        cpt_out1 = cpt_conv1

                
        print('***** pre_F : ',pre_F       )
        print('***** conv_F: ',conv_F      )
        print('***** R_N   : ',R_N         )
        
        print('***** plus  : ',plus*is_plus)
        print('***** rate  : ',rate*is_plus)
        
        print('***** P_N   : ',P_N*is_pt   )
        print('***** T_F   : ',T_F*is_pt   )
        print('***** PT_F  : ',PT_F*is_pt*isPT_F )            
        print('***** T     : ',T           ) 
        
        print('***** drop  : ',drop        )
        
        print(cpt_out1.shape)
        
        Y = cpt_out1
        Y = tf.reshape(tf.transpose(Y, (0, 2, 3, 1)), (-1, self.H_height*self.H_width, self.out_dim))
        Y = self.converter_mat.T @ Y
        return Y



# from __future__ import print_function
from tensorflow.keras.layers import (
    Input,
    Activation,
    # merge,
    add,
    Dense,
    Reshape,
    BatchNormalization
)
from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model
#from keras.utils.visualize_util import plot


def _shortcut(input, residual):
    # return merge([input, residual], mode='sum')
    return add([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        # return Conv2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
        
        return Conv2D(filters=nb_filter, kernel_size=(nb_row,nb_col), strides=subsample, padding="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


class Conv2DFilterPool(tf.keras.layers.Layer):
    def __init__(self, num_outputs, use_bias=True, padding="same"):
        super(Conv2DFilterPool, self).__init__()
        self.num_outputs = num_outputs
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_shape):
        if self.use_bias:
            self.offset = self.add_weight(
                shape=[1,1,1,self.num_outputs], initializer="random_normal", 
                trainable=True, dtype=tf.float32, name='offset'
            )
        self.hex_filter = tf.constant( np.array([
                [ 0,  1,  0],
                [ 1,  0,  1],
                [ 0,  1,  0],
                [ 1,  0,  1],
                [ 0,  1,  0]
            ]).reshape([5,3,1,1]) , dtype=tf.float32 )
        self.kernel = self.add_weight(
            shape=[5,3,int(input_shape[-1]), self.num_outputs], 
            initializer=tf.keras.initializers.RandomNormal(stddev=1./7.), #"random_normal", 
            trainable=True, dtype=tf.float32, name='kernel'
        ) * self.hex_filter

    def call(self, input):
        conv = tf.keras.backend.conv2d(input, self.kernel * self.hex_filter, 
                            data_format="channels_last", padding=self.padding)
        if self.use_bias:
            return tf.nn.tanh( conv + self.offset )
        else:
            return tf.nn.tanh( conv )


class MySTResNet():
    def __init__(self, extdata, args, out_dim=2):
        from tensorflow.keras import backend as K
        K.set_image_data_format('channels_first')
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.K = args.K
        self.num_K = args.num_K
        self.out_dim = out_dim
        self.adj_mats = []
        self.ext_feats = []
        self.poi_local = extdata['LU_TY']

        if args.node != None:
            for node in args.node:
                self.ext_feats.append(tf.cast(extdata[node] / (extdata[node].max(0)+1e-10), tf.float32))

        if args.edge != None:
            for edge in args.edge:
                self.adj_mats.append(row_normalize(extdata[edge]))

        self.H_width, self.H_height = extdata['H_width'], extdata['H_height']
        midx = [y*self.H_width + x for x,y in extdata['HXYS']]
        self.converter_mat = np.eye(self.H_width*self.H_height)[midx].T
        

    def __call__(self, kwargs):
        XC, XP, XQ = kwargs['XC'], kwargs['XP'], kwargs['XQ']
        WC, WP, WQ = kwargs['WC'], kwargs['WP'], kwargs['WQ']
        TEC, TEP, TEQ, TEY = kwargs['TEC'], kwargs['TEP'], kwargs['TEQ'], kwargs['TEY']
        
        TEC = tf.cast(TEC, tf.int32)
        TEP = tf.cast(TEP, tf.int32)
        TEQ = tf.cast(TEQ, tf.int32)
        TEY = tf.cast(TEY, tf.int32)
        C_num = TEC.shape[1]
        P_num = TEP.shape[1]
        Q_num = TEQ.shape[1]
        print(C_num, P_num, Q_num)

        TE = tf.cast(TEY, tf.int32)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.expand_dims(TE, -1)
        TE = tf.tile(TE, (1, 1, self.H_height, self.H_width))

        XC = tf.reshape((self.converter_mat @ XC), (-1, C_num, self.H_height, self.H_width, 2))
        XP = tf.reshape((self.converter_mat @ XP), (-1, P_num, self.H_height, self.H_width, 2))
        XQ = tf.reshape((self.converter_mat @ XQ), (-1, Q_num, self.H_height, self.H_width, 2))
        
        XC = tf.reshape(tf.transpose(XC, (0, 1, 4, 2, 3)), (-1, C_num, 2, self.H_height, self.H_width))
        XP = tf.reshape(tf.transpose(XP, (0, 1, 4, 2, 3)), (-1, P_num, 2, self.H_height, self.H_width))
        XQ = tf.reshape(tf.transpose(XQ, (0, 1, 4, 2, 3)), (-1, Q_num, 2, self.H_height, self.H_width))

        # XC = XC[:, -3:, ...]
        # XP = XP[:, -3:, ...]
        # XQ = XQ[:, -3:, ...]

        # c_conf=XC.shape[1:]
        # p_conf=XP.shape[1:]
        # t_conf=XQ.shape[1:]
        external_dim=8
        nb_residual_unit=3
        CF=64
        nb_flow_out = 2

        # main input
        main_inputs = []
        outputs = []
        # for conf, input in zip([c_conf, p_conf, t_conf], [XC, XP, XQ]):
        for inp in [XC, XP, XQ]:
            # if conf is not None:
            len_seq, nb_flow, map_height, map_width = inp.shape[1:]
            inp = tf.reshape(inp, (-1, nb_flow * len_seq, map_height, map_width))
            # input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            # main_inputs.append(input)
            # Conv1
            conv1 = Conv2D(
                filters=CF, kernel_size=(5, 3), padding="same")(inp)
            # inp = tf.transpose(inp, (0, 2, 3, 1))
            # conv1 = Conv2DFilterPool(CF)(inp)
            # conv1 = tf.transpose(conv1, (0, 3, 1, 2))

            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=CF,
                            repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Conv2D(
                filters=nb_flow_out, kernel_size=(5,3), padding="same")(activation)
            
            # activation = tf.transpose(activation, (0, 2, 3, 1))
            # conv2 = Conv2DFilterPool(nb_flow_out)(activation)
            # conv2 = tf.transpose(conv2, (0, 3, 1, 2))
            outputs.append(conv2)

        # parameter-matrix-based fusion
        if len(outputs) == 1:
            main_output = outputs[0]
        else:
            from DST_network_ilayer import iLayer
            new_outputs = []
            for output in outputs:
                print('output', output.shape)
                n_output = iLayer()(output)
                # n_output = output
                new_outputs.append(n_output)
                print('n_output', n_output.shape)
            # main_output = merge(new_outputs, mode='sum')
            main_output = add(new_outputs)

        # # fusing with external component
        # if external_dim != None and external_dim > 0:
        #     # external input
        #     external_input = Input(shape=(external_dim,))
        #     main_inputs.append(external_input)
        #     embedding = Dense(10)(external_input)
        #     embedding = Activation('relu')(embedding)
        #     h1 = Dense(nb_flow_out * map_height * map_width)(embedding)
        #     activation = Activation('relu')(h1)
        #     external_output = Reshape((nb_flow_out, map_height, map_width))(activation)
        #     # main_output = merge([main_output, external_output], mode='sum')
        #     print('main,external', main_output.shape, external_output.shape)
        #     main_output = add([main_output, external_output])
        # else:
        #     print('external_dim:', external_dim)

        # main_output = Activation('tanh')(main_output)
        # model = Model(main_inputs, main_output)

        print(main_output.shape)
        Y = main_output
        Y = tf.reshape(tf.transpose(Y, (0, 2, 3, 1)), (-1, self.H_height*self.H_width, self.out_dim))
        Y = self.converter_mat.T @ Y
        print('Y', Y.shape)
        return Y



####################################################
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def ModelSet(model_name, extdata, args, **kwargs):
    model = str_to_class(model_name)(extdata, args)
    return (model(kwargs) ) * extdata['max_values'] # +0.5
