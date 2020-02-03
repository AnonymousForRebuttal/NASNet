import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from torchvision.utils import make_grid
import time,math,os
import numpy as np
from torch.backends import cudnn
from torch import optim
import matplotlib.pyplot as plt
import torch,warnings
from torch import nn
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')


parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=300000)
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--save_step',type=int,default=1000)
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='trained_models')
parser.add_argument('--trainset',type=str,default='Rain100H')
parser.add_argument('--task',type=str,default='streak')
parser.add_argument('--net',type=str,default='nas')
parser.add_argument('--ss',type=int,default=3,help='stages')
parser.add_argument('--blocks',type=int,default=18,help='residual_broups')
parser.add_argument('--bs',type=int,default=2,help='batch size')
parser.add_argument('--crop_size',type=int,default=240,help='crop_size for lr')
parser.add_argument('--ssimloss',action='store_true',help='ssim loss for train')
parser.add_argument('--threads',type=int,default=0,help='for loading data')
opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
cwd='/'.join(os.path.realpath(__file__).split('/')[:-2])
model_name=opt.trainset+'_'+opt.task+'_'+opt.net.split('.')[0]+'_'+str(opt.ss)+'_'+str(opt.blocks)
if opt.ssimloss:
	model_name=model_name+'_ssimloss'
opt.model_dir=os.path.join(cwd,'net',opt.model_dir,model_name+'.pk')
print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]) )
print('model_name:',model_name)


def tensorShow(r,c,tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(r,c,1+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

mean_opt={
	'MPIDstreak':[0.47748442, 0.44561987 ,0.40421578],
	'MPIDdrop':[0.50788142 ,0.5181989 , 0.484861  ],
	'MPIDmist':[0.39551503, 0.3972878 , 0.3676113 ],
	'SPA_REALstreak':[0.37963131 ,0.37317694 ,0.34395947],
	'Rain100Hstreak':[0.28889797 ,0.29333464 ,0.26457639],
	'Rain100Lstreak':[0.20238657, 0.20641472 ,0.17443697],
	'RainCityScapesrainfog':[0.39394617, 0.44335372 ,0.40406548],
	
}

std_opt={
	'MPIDstreak': [0.24147542,0.23230714 ,0.23012931],
	'MPIDdrop':[0.18620518 ,0.18312409, 0.19519102],
	'MPIDmist':[0.14311445 ,0.13827545 ,0.1329295 ],
	'SPA_REALstreak':[0.14067527, 0.13811357 ,0.1289146 ],
	'Rain100Hstreak':[0.35875455, 0.36073939, 0.33314634],
	'Rain100Lstreak':[0.27103779 ,0.2704343 , 0.23743652],
	'RainCityScapesrainfog': [0.14550411, 0.15230632 ,0.15129043],
}