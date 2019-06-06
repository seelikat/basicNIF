import os, glob, pdb
import numpy as np
from scipy.io import loadmat, savemat

import chainer as cn
import chainer.links as L
import chainer.functions as F
import chainer.initializers as I
from chainer.training import extensions
from chainer.dataset import concat_examples

from args import BasicConfig as cfg
from utils import PlotLearnedParams

from sklearn.preprocessing import StandardScaler


'''
This example should illustrate how NIF modeling is implemented on a small  
data set (360 handwritten letter images, V1 and V2 only) that could be 
included in the same code package. The model does not require a GPU and 
can run on a notebook. 

The learned channels and a few of the voxel-wise receptive fields will be
written into the current directory after every epoch. 

$ pip install chainer
$ python train_model.py
'''

class LetterData(cn.dataset.DatasetMixin):

    def __init__(self, stim, bold, rois):

        self.stim = stim
        self.bold = bold
        self.rois = rois

        if cfg.standardize: 
            for roi in self.rois: 
                scaler = StandardScaler()
                scaler.fit(self.bold[roi])
                self.bold[roi] = scaler.transform(self.bold[roi])

    def __len__(self):
        return self.stim.shape[0]

    def get_example(self, i):

        datadict = { roi : self.bold[roi][i,:].astype(cfg.dtype) for roi in self.rois }
        datadict['x'] = self.stim[i].astype(cfg.dtype)[np.newaxis,:,:]

        return datadict


class V1V2Model(cn.Chain): 

    def __init__(self, roisz):
        super(V1V2Model, self).__init__()

        with self.init_scope(): 

            # Area-specific layers
            self.toV1  = L.ConvolutionND(ndim=2, in_channels=None, out_channels=cfg.n_c['V1'], ksize=3, pad=1, 
                                            initialW = I.HeNormal(scale=1.0/cfg.n_c['V1']) )
            self.toV2  = L.ConvolutionND(ndim=2, in_channels=None, out_channels=cfg.n_c['V2'], ksize=3, pad=1, 
                                            initialW = I.HeNormal(scale=1.0/cfg.n_c['V2']) )

            # Observation models
            roi = 'V1'
            self.Uc_V1 = cn.Parameter( I.HeNormal(), shape=[cfg.n_c['V1'], roisz[roi]], name='Uc_V1')
            
            self.Ux_V1_ks = cn.Parameter( I.HeNormal(), shape=[roisz[roi], cfg.indim/(2**1), cfg.rank], name='Ux_V1_ks')
            self.Uy_V1_ks = cn.Parameter( I.HeNormal(), shape=[roisz[roi], cfg.indim/(2**1), cfg.rank], name='Uy_V1_ks')
            self.Am_V1_ks = cn.Parameter( I.HeNormal(), shape=[roisz[roi], cfg.rank], name='Am_V1_ks')
            self.bias_V1  = L.Bias(shape=[roisz[roi]])

            roi = 'V2'
            self.Uc_V2 = cn.Parameter( I.HeNormal(), shape=[cfg.n_c['V2'], roisz[roi]], name='Uc_V2')

            self.Ux_V2_ks = cn.Parameter( I.HeNormal(), shape=[roisz[roi], cfg.indim/(2**2), cfg.rank], name='Ux_V2_ks')
            self.Uy_V2_ks = cn.Parameter( I.HeNormal(), shape=[roisz[roi], cfg.indim/(2**2), cfg.rank], name='Uy_V2_ks')
            self.Am_V2_ks = cn.Parameter( I.HeNormal(), shape=[roisz[roi], cfg.rank], name='Am_V2_ks')
            self.bias_V2  = L.Bias(shape=[roisz[roi]])


    def rank1_observe(self, roi, U, Uc, Ux, Uy, bias): 

        U_xyc = F.transpose( U, axes=[0,3,2,1] )        # [b,s,s,c] 
        U_xyv = F.matmul( U_xyc, Uc )                   # [b,s,s,v] 

        U_xvy = F.transpose( U_xyv, axes=[0,1,3,2] )    # [b,s,v,s]
        U_xv = F.sum( Uy * U_xvy , axis=3 )   

        U_vx = F.transpose( U_xv, axes=[0,2,1] )        # [b,v,s]
        obs = bias( F.sum( Ux * U_vx , axis=2 ) )

        return obs


    def rankn_observe(self, roi, U, Uc, Ux_ks, Uy_ks, Am_ks, bias): 

        Ux_ks = F.softmax(Ux_ks, axis=1) 
        Uy_ks = F.softmax(Uy_ks, axis=1)  # positivity constraint & slight denoise

        obs = 0
        for k in range(cfg.rank): 
            obs += F.softplus(Am_ks[:,k]) * self.rank1_observe(roi, U, Uc, Ux_ks[:,:,k], Uy_ks[:,:,k], bias) 

        return obs


    def forward(self, input): 

        obs = {}

        # forward pass
        u_v1  = F.average_pooling_nd( F.sigmoid( self.toV1(  input ) ), ksize=2 )
        u_v2  = F.average_pooling_nd( F.sigmoid( self.toV2(  u_v1  ) ), ksize=2 )

        # observation (factorize tensors u)
        obs['V1'] = self.rankn_observe('V1', u_v1, self.Uc_V1, self.Ux_V1_ks, self.Uy_V1_ks, self.Am_V1_ks, self.bias_V1 )
        obs['V2'] = self.rankn_observe('V2', u_v2, self.Uc_V2, self.Ux_V2_ks, self.Uy_V2_ks, self.Am_V2_ks, self.bias_V2 )

        return obs


class Regressor(cn.Chain):

    def __init__(self, predictor):
        super(Regressor, self).__init__(predictor=predictor)

    def forward(self, **datadict):

        obs = self.predictor(datadict['x'])

        loss = 0
        for roi in obs.keys(): 
            loss += F.mean_squared_error(datadict[roi], obs[roi])

        cn.report({'loss': loss}, self)

        return loss


if __name__ == "__main__":

    rois =  ['V1', 'V2']

    ### Build training data ###
    stim = loadmat('stim.mat')['stim']
    bold = loadmat('bold.mat')

    bold  = { 'V1':bold['V1'], 'V2':bold['V2'] }

    train = LetterData(stim, bold, rois)
    train_iter = cn.iterators.SerialIterator( train , cfg.nbatch ) 

    ### Set up model ###
    roisz = { 'V1':bold['V1'].shape[1] , 'V2':bold['V2'].shape[1] }
    
    model = Regressor( V1V2Model(roisz) )
    optimizer = cn.optimizers.Adam(alpha=cfg.lr_alpha)
    optimizer.setup(model)
    updater = cn.training.StandardUpdater(train_iter, optimizer, device=cfg.gpuid )

    ### Train model ###
    trainer = cn.training.Trainer(updater, (cfg.nepochs, 'epoch'), cfg.basedir)

    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss']), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport( (100, 'iteration', 'main/loss') ) )
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.snapshot_object(model, 'snapV1V2Model_{.updater.iteration}'), trigger=(5, 'epoch'))
    trainer.extend(PlotLearnedParams( model, gpuid=cfg.gpuid, rois=rois, rank=cfg.rank), trigger=(1, 'epoch'))

    trainer.run()