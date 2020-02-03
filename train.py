import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math,warnings
import numpy as np
from torch.backends import cudnn
from radam import RAdam
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')
from option import opt,model_name,cwd
from data_utils import *
from ssim_loss import SSIM as ssimloss
warnings.filterwarnings('ignore')

models_={
	'nas':NASNet(ss=opt.ss,blocks=opt.blocks)
}
loaders_={
	'train':train_loader,
	'test':test_loader,
}

T=opt.steps	

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,optim,criterion):
	start_step=0
	if os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		net.load_state_dict(ckp['model'])
		start_step=ckp['step']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=opt.lr
		lr=lr_schedule_cosdecay(step,T)
		for param_group in optim.param_groups:
			param_group["lr"] = lr  
		x,y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device)
		out=net(x)
		loss=criterion[0](out,y)
		if opt.ssimloss:
			loss3=criterion[-1](out,y)
			loss=loss+(1-loss3)
		loss.backward()
		optim.step()
		optim.zero_grad()
		print(f'\r train loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f}',end='',flush=True)
		if step % opt.save_step ==0 :
			torch.save({
						'step':step,
						'model':net.state_dict()
			},opt.model_dir)
			print(f'\n model saved at step :{step}')




if __name__ == "__main__":
	loader_train=loaders_['train']
	net=models_[opt.net]
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.ssimloss:
		criterion.append(ssimloss().to(opt.device))
	optimizer = RAdam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,optimizer,criterion)
	

