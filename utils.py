import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.io import savemat
from scipy.misc import imsave

import chainer.functions as F
from chainer.training import Extension
from chainer.training.extensions.evaluator import Evaluator
from chainer import reporter as reporter_module
from chainer.variable import Variable



class PlotLearnedParams(Extension):

    def __init__(self, model, gpuid, rois, rank):
        super(PlotLearnedParams, self).__init__()

        self.model = model
        self.gpuid = gpuid
        self.rois = rois
        self.rank = rank


    def __call__(self, trainer=None):

        if hasattr(self, 'name'):
            prefix = self.name + '/'
        else:
            prefix = ''

        self.plotparams()


    def finalize(self): 
        self.plotparams()


    def plotparams(self):
    
        ### Extract parameters from current model ###
        allparams = {} ; convWs = {}

        tocpuext = '' if self.gpuid==-1 else '.get()'

        for roi in self.rois: 
            allparams['convW_'+roi] = eval( "self.model.predictor.to"+roi+".W.data"+tocpuext )

        for roi in self.rois: 
            allparams['Uc_' + roi] = eval("self.model.predictor.Uc_"+roi+".data" + tocpuext )
            allparams['Ux_'+roi+'_ks'] = eval("self.model.predictor.Ux_"+roi+"_ks.data" + tocpuext )
            allparams['Uy_'+roi+'_ks'] = eval("self.model.predictor.Uy_"+roi+"_ks.data" + tocpuext )
            allparams['Am_'+roi+'_ks'] = eval("self.model.predictor.Am_"+roi+"_ks.data" + tocpuext )

        savemat('modelparams.mat', allparams ) 

        ### Plot a few receptive fields ###
        for roi, vox_ids in {'V1':[612, 761, 494, 536, 219], 'V2':[1477, 289, 648, 292, 1151]}.items(): 
            # (plot 5 well-explained voxels for V1 and V2)
            for vox_id in vox_ids: 
                plt.figure()

                Am = F.softplus(allparams['Am_'+roi+'_ks'])
                Us = 0.0
                for r in range(self.rank):     
                    Us += Am[vox_id,r].data * np.outer(  F.softmax(allparams['Ux_'+roi+'_ks'][vox_id,:,r], axis=0).data, 
                                                         F.softmax(allparams['Uy_'+roi+'_ks'][vox_id,:,r], axis=0).data  )
                Us = zoom(Us, order=0, zoom=56.0/Us.shape[0] )

                lim = np.abs(Us).max()
                plt.imshow(Us, cmap='coolwarm', vmin=-lim, vmax=lim)
                plt.xticks([]) ; plt.yticks([])
                plt.savefig('Us_'+roi+'_vox'+str(vox_id)+'.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close()


        ### Write learned weights for V1 ###
        roi = 'V1'
        convW = np.tile(allparams['convW_'+roi][:,np.newaxis,:,:,:], reps=[1,3,1,1,1]).transpose(0,2,3,4,1) # (imsave needs RGB)

        for ch_idx in range(convW.shape[0]):
            chan = np.squeeze( convW[ch_idx] ) 
            imsave(roi+'_chan'+str(ch_idx)+'.png', chan) 
