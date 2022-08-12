import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class STEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.D = D

    def build(self, input_shape):
        # self.SE = self.add_weight(
        #     shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        # )
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TEC, TEP, TEQ):
        # spatial embedding
        # SE = tf.expand_dims(tf.expand_dims(self.SE, axis = 0), axis = 0)
        # TEC temporal embedding
        
        num_c = TEC.shape[-2]
        num_p = TEP.shape[-2]
        num_q = TEQ.shape[-2] 
        
        TE = tf.concat((TEC, TEP, TEQ), -2)
        
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        
        STE = SE + TE
        STE_C = STE[:, :num_c, ...]
        STE_P = STE[:, num_c:num_c+num_p, ...]
        STE_Q = STE[:, num_c+num_p:, ...]
        
        return STE_C, STE_P, STE_Q
    

class STEWmbedding(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STEWmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.D = D

    def build(self, input_shape):
        # self.SE = self.add_weight(
        #     shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        # )
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D)])
        self.FC_WE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TEC, TEP, TEQ, WC, WP, WQ):
        # spatial embedding
        # SE = tf.expand_dims(tf.expand_dims(self.SE, axis = 0), axis = 0)
        # TEC temporal embedding
        
        num_c = TEC.shape[-2]
        num_p = TEP.shape[-2]
        num_q = TEQ.shape[-2] 
        
        TE = tf.concat((TEC, TEP, TEQ), -2)
        WE = tf.concat((WC, WP, WQ), -2)
        
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        WE = tf.expand_dims(WE, axis = 2)
        # TE = self.FC_WE(tf.concat((self.FC_TE(TE), WE), -1))
        TE = self.FC_TE(TE) + self.FC_WE(WE)
        
        
        STE = SE + TE
        STE_C = STE[:, :num_c, ...]
        STE_P = STE[:, num_c:num_c+num_p, ...]
        STE_Q = STE[:, num_c+num_p:, ...]
        
        return STE_C, STE_P, STE_Q
    

class STEmbedding_Y(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STEmbedding_Y, self).__init__()
        self.num_nodes = num_nodes
        self.D = D

    def build(self, input_shape):
        # self.SE = self.add_weight(
        #     shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
        # )
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TEC, TEP, TEQ, TEY):
        # spatial embedding
        # SE = tf.expand_dims(tf.expand_dims(self.SE, axis = 0), axis = 0)
        TEY = tf.expand_dims(TEY, axis=1)
        # TEC temporal embedding
        
        num_c = TEC.shape[-2]
        num_p = TEP.shape[-2]
        num_q = TEQ.shape[-2]
        num_y = TEY.shape[-2]
        
        TE = tf.concat((TEC, TEP, TEQ, TEY), -2)
        
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        minuteofday = tf.one_hot(TE[..., 2], depth = 4)
        holiday = tf.one_hot(TE[..., 3], depth = 1)
        TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        
        STE = SE + TE
        STE_C = STE[:, : num_c, ...]
        STE_P = STE[:, num_c : num_c+num_p, ...]
        STE_Q = STE[:, num_c+num_p : num_c+num_p+num_q, ...]
        STE_Y = STE[:, num_c+num_p+num_q : , ...]
        
        return STE_C, STE_P, STE_Q, STE_Y
    
    
# class STEmbedding_Y(tf.keras.layers.Layer):
#     def __init__(self, num_nodes, D):
#         super(STEmbedding_Y, self).__init__()
#         self.num_nodes = num_nodes
#         self.D = D
#         print(num_nodes, D)

#     def build(self, input_shape):
#         self.SE = self.add_weight(
#             shape=(self.num_nodes, self.D), initializer="random_normal", trainable=True, dtype=tf.float32, name='SE'
#         )
#         self.FC_TEC = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D),])
#         self.FC_TEP = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D),])
#         self.FC_TEQ = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D),])
#         self.FC_TEY = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D),])
        
#     def call(self, TEC, TEP, TEQ, TEY):
#         # spatial embedding
#         SE = tf.expand_dims(tf.expand_dims(self.SE, axis = 0), axis = 0)
#         # SE = self.FC1(SE)
#         # TEC temporal embedding
#         dayofweek = tf.one_hot(TEC[..., 0], depth = 7)
#         timeofday = tf.one_hot(TEC[..., 1], depth = 24)
#         minuteofday = tf.one_hot(TEC[..., 2], depth = 4)
#         TEC = tf.concat((dayofweek, timeofday, minuteofday), axis = -1)
#         TEC = tf.expand_dims(TEC, axis = 2)
#         TEC = self.FC_TEC(TEC)
        
#         # TEP temporal embedding
#         dayofweek = tf.one_hot(TEP[..., 0], depth = 7)
#         timeofday = tf.one_hot(TEP[..., 1], depth = 24)
#         minuteofday = tf.one_hot(TEP[..., 2], depth = 4)
#         TEP = tf.concat((dayofweek, timeofday, minuteofday), axis = -1)
#         TEP = tf.expand_dims(TEP, axis = 2)
#         TEP = self.FC_TEP(TEP)
        
#         # TEQ temporal embedding
#         dayofweek = tf.one_hot(TEQ[..., 0], depth = 7)
#         timeofday = tf.one_hot(TEQ[..., 1], depth = 24)
#         minuteofday = tf.one_hot(TEQ[..., 2], depth = 4)
#         TEQ = tf.concat((dayofweek, timeofday, minuteofday), axis = -1)
#         TEQ = tf.expand_dims(TEQ, axis = 2)
#         TEQ = self.FC_TEQ(TEQ)
        
#         # TEQ temporal embedding
#         dayofweek = tf.one_hot(TEY[..., 0], depth = 7)
#         timeofday = tf.one_hot(TEY[..., 1], depth = 24)
#         minuteofday = tf.one_hot(TEY[..., 2], depth = 4)
#         TEY = tf.concat((dayofweek, timeofday, minuteofday), axis = -1)
#         TEY = tf.expand_dims(TEY, axis = 2)
#         TEY = self.FC_TEY(TEY)
        
#         return tf.add(SE, TEC), tf.add(SE, TEP), tf.add(SE, TEQ), tf.add(SE, TEY)

    
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, K, d, use_mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.use_mask = use_mask

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        if self.use_mask:
            batch_size = tf.shape(X)[0]
            num_step = X.get_shape()[1].value
            N = X.get_shape()[2].value
            mask = tf.ones(shape = (num_step, num_step))
            mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
            mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
            mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
            mask = tf.cast(mask, dtype = tf.bool)
            attention = tf.compat.v2.where(
                condition = mask, x = attention, y = -2 ** 15 + 1)
            
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
    
class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FC_S = keras.Sequential([
            layers.Dense(self.D, use_bias=False),])
        self.FC_T = keras.Sequential([
            layers.Dense(self.D),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.D),])
        
    def call(self, HS, HT):
        XS = self.FC_S(HS)
        XT = self.FC_T(HT)
        
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC_H(H)
        return H
    
class CPTFusion(tf.keras.layers.Layer):
    def __init__(self, D, out_dim):
        super(CPTFusion, self).__init__()
        self.D = D
        self.out_dim = out_dim

    def build(self, input_shape):
        self.FC_C = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_P = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_Q = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.out_dim),])
        
    def call(self, XC, XP, XQ):
        ZC = self.FC_C(XC)
        ZP = self.FC_P(XP)
        ZQ = self.FC_Q(XQ)

        Z = tf.concat((ZC, ZP, ZQ), -1)
        Z = tf.nn.softmax(Z)
        return self.FC_H(Z[..., 0:1] * XC + Z[..., 1:2] * XP + Z[..., 2:] * XQ)

        
class MultiFusion(tf.keras.layers.Layer):
    def __init__(self, D, out_dim):
        super(MultiFusion, self).__init__()
        self.D = D
        self.out_dim = out_dim

    def build(self, input_shape):
        self.FC_X = [keras.Sequential([
            layers.Dense(1, use_bias=False),]) for _ in range(input_shape[-1])]
        # self.FC_H = keras.Sequential([
            # layers.Dense(self.D, activation='relu'),
            # layers.Dense(self.out_dim),])
        
    def call(self, Xs):
        print('Xs.shape', Xs.shape)
        Zs = []
        for i in range(Xs.shape[-1]):
            X = Xs[..., i]
            Z = self.FC_X[i](X)
            Zs.append(Z)

        # ZC = self.FC_C(XC)
        # ZP = self.FC_P(XP)
        # ZQ = self.FC_Q(XQ)

        Z = tf.stack((Zs), -1)
        Z = tf.expand_dims(tf.nn.softmax(Z), -2)
        print('Z.shape', Z.shape)

        return tf.reduce_sum(Z * Xs, -1)

        # return self.FC_H(Z[..., 0:1] * XC + Z[..., 1:2] * XP + Z[..., 2:] * XQ)



class GSTAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(GSTAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.SA_layer = SpatialAttention(self.K, self.d)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.SA_layer(X, STE)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H
    
class SGTAblock(tf.keras.layers.Layer):
    def __init__(self, K, d, adj_mats, ext_feats):
        super(SGTAblock, self).__init__()
        self.K = K
        self.d = d
        self.adj_mats = adj_mats
        self.ext_feats = ext_feats
        self.D = K*d

    def build(self, input_shape):
        self.SA_layer = SpatialConv(self.D, self.adj_mats, self.ext_feats)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        HS = self.SA_layer(X)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H
    
# def GSTAttBlock(X, STE, K, d, bn, bn_decay, is_training):
#     HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
#     HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training)
#     H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
#     return tf.add(X, H)
    
# def RSTAttBlock(Z, STEZ, ADJ_DY, K, d, bn, bn_decay, is_training, args):
#     D = K*d
#     # HS = spatialAttention(Z, STEZ, K, d, bn, bn_decay, is_training)
#     HT = temporalAttention(Z, STEZ, K, d, bn, bn_decay, is_training)
#     # H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
#     return tf.add(Z, HT)

# def RSTAttBlock(Z, STEZ, ADJ_DY, K, d, bn, bn_decay, is_training, args):
#     D = K*d
#     num_step = tf.shape(Z)[1]

#     ZS = DyConv(ADJ_DY, Z, num_step, D)
#     ZS = tf_utils.batch_norm(ZS, is_training = is_training, bn_decay = bn_decay)
#     ZT = temporalAttention(Z, STEZ, K, d, bn, bn_decay, is_training, mask=True)
#     H = gatedFusion(ZS, ZT, K * d, bn, bn_decay, is_training)

#     return tf.add(Z, H)
    
class RSTAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(RSTAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        #self.SA_layer = spatialAttention(self.K, self.d)
        self.TA_layer = temporalAttention(self.K, self.d)
        self.GF = gatedFusion(self.D)
        
    def call(self, Z, STEZ):
        K = self.K
        d = self.d
        
        #HS = self.SA_layer(Z, STEZ)
        HT = self.TA_layer(Z, STEZ)
        #H = self.GF(HS, HT)
        return Z + HT
    
    
class TransformAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D)])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
    
class SpatialConv(tf.keras.layers.Layer):
    def __init__(self, D, adj_mats, ext_feats):
        super(SpatialConv, self).__init__()
        self.D = D
        self.adj_mats = adj_mats
        self.ext_feats = ext_feats

    def build(self, input_shape):
        self.FC = layers.Dense(self.D, activation="relu")
        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.D, activation='sigmoid')]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.D, activation='sigmoid')]) for i in range(len(self.ext_feats))]
        
    def call(self, x):
        x0 = x
        for i, support in enumerate(self.adj_mats):
            # premultiply the concatened inputs and state with support matrices
            x_support = support@x0
            x = tf.concat([x, x_support], axis=-1)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
             
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.D**0.5 # @ tf.transpose(FE)
            support = tf.nn.softmax(support)
            
            x_support = support@x0
            x = tf.concat([x, x_support], axis=-1)

        x = self.FC(x)
        return x
    
    
    
    
# class MyDenseLayer(tf.keras.layers.Layer):
#     def __init__(self, SE, SEZ, args):
#         super(MyDenseLayer, self).__init__()
#         self.SE = SE
#         self.SEZ = SEZ
#         self.K = args.K
#         self.d = args.d
#         self.D = args.K * args.d
#         self.P = args.P
#         self.Q = args.Q
#         self.L = args.L
#         self.LZ = args.LZ

#     def build(self, input_shape):
#         self.STE_layer = STEmbedding(D)
#         self.FC1 = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(D)])
#         self.FC2 = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(1)])
#         self.FCZ1 = keras.Sequential([
#                         layers.Dense(D, activation="relu"),
#                         layers.Dense(D)])
#         #self.FCZ2 = keras.Sequential([
#         #                layers.Dense(D, activation="relu"),
#         #                layers.Dense(1)])
#         self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
#         self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
#         self.RSTA_enc = [RSTAttBlock(self.K, self.d) for _ in range(self.LZ)]
#         self.RSTA_dec = [RSTAttBlock(self.K, self.d) for _ in range(self.LZ)]
#         self.trans_layer = transformAttention(self.K, self.d)
#         self.trans_layer_Z = transformAttention(self.K, self.d)
#         self.bipartite_conv_P = bipartiteConv(self.D)
#         self.bipartite_conv_Q = bipartiteConv(self.D)

#     def call(self, X, Z, TE):
#         STE, STEZ = self.STE_layer(self.SE, self.SEZ, TE)
#         STE_P  = STE[:, : self.P]
#         STE_Q  = STE[:, self.P :]
#         STEZ_P = STEZ[:, : self.P]
#         STEZ_Q = STEZ[:, self.P :]
        
        
#         Z = tf.expand_dims(Z, axis = -1)
#         Z = self.FCZ1(Z)
#         # encoder
#         for i in range(self.LZ):
#             Z = self.RSTA_enc[i](Z, STEZ_P)
#         # transAtt
#         ZP = Z
#         Z = self.trans_layer_Z(Z, STEZ_P, STEZ_Q)
#         # decoder
#         for i in range(self.LZ):
#             Z = self.RSTA_dec[i](Z, STEZ_Q)
#         # output
#         ZQ = Z
        
#         XZP = self.bipartite_conv_P(ZP, ADJ_GR)
#         XZQ = self.bipartite_conv_Q(ZQ, ADJ_GR)
        
        
#         X = tf.expand_dims(X, axis = -1)
#         X = self.FC1(X)
#         # encoder
#         for i in range(self.L):
#             X = self.GSTA_enc[i](X, STE_P)
#         # transAtt
#         X = self.trans_layer(X, XZP, XZQ)
#         # decoder
#         for i in range(self.L):
#             X = self.GSTA_dec[i](X, STE_Q)
#         # output
#         X = self.FC2(X)
#         return tf.squeeze(X, axis = 3)

