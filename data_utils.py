import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random , glob
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt,cwd,mean_opt,std_opt

def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(211+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

class Rain_Dataset(data.Dataset):
    def __init__(self,path,size=240,task='drop',dataset='MPID',mode='train'):
        super(Rain_Dataset,self).__init__()
        self.mean=mean_opt[dataset+task]
        self.std=std_opt[dataset+task]
        self.des=dataset+task
        self.size=size
        self.mode=mode
        self.rain_img_dirs=os.listdir(os.path.join(path,dataset,task,f'{mode}/data'))
        self.rain_imgs=[os.path.join(path,dataset,task,f'{mode}/data/',img) for img in self.rain_img_dirs]
        self.clean_dir=os.path.join(path,dataset,task,f'{mode}/gt/')
    def __getitem__(self, index):
        rain=Image.open(self.rain_imgs[index])
        img=self.rain_img_dirs[index]
        if self.des=='MPIDmist':
            id=img.split('/')[-1].split('.')[0]
        else :
            id=img.split('/')[-1].split('_')[0]
        try :
            clean_name=id+f'_clean'
            clean=Image.open(glob.glob(self.clean_dir+clean_name+'.*')[0])
        except Exception:
            clean_name=id
            clean=Image.open(glob.glob(self.clean_dir+clean_name+'.*')[0])
        if rain.size != clean.size :
            print(id,'------------------------------------')
        if self.mode=='train':
            i,j,h,w=tfs.RandomCrop.get_params(clean,output_size=(self.size,self.size))
            rain=FF.crop(rain,i,j,h,w)
            clean=FF.crop(clean,i,j,h,w)
            rain,clean=self.augData(rain.convert('RGB'),clean.convert('RGB'))
        else :
            #for test all-img
            rain=tfs.ToTensor()(rain)
            clean=tfs.ToTensor()(clean)
            rain=tfs.Normalize(mean=self.mean,std=self.std)(rain)
        return rain,clean
    def augData(self,data,target):
        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        data=tfs.RandomHorizontalFlip(rand_hor)(data)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data=FF.rotate(data,90*rand_rot)
            target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=self.mean,std=self.std)(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.rain_img_dirs)

class Rain_Dataset_SPAREAL(data.Dataset):
    def __init__(self,path,size=240,dataset='SPA_REAL',mode='train'):
        super(Rain_Dataset_SPAREAL,self).__init__()
        self.mean=mean_opt[dataset+'streak']
        self.std=std_opt[dataset+'streak']
        self.size=size
        self.mode=mode
        self.rain_img_dirs=os.listdir(os.path.join(path,dataset,f'{mode}/data'))
        self.rain_imgs=[os.path.join(path,dataset,f'{mode}/data/',img) for img in self.rain_img_dirs]
        self.clean_dir=os.path.join(path,dataset,f'{mode}/gt/')
    def __getitem__(self, index):
        rain=Image.open(self.rain_imgs[index])
        img=self.rain_img_dirs[index]
        if self.mode=='train':
            id=img.split('/')[-1].split('-')
            id2=id[1].split('_')
            clean_name=id[0]+'_'+id2[1]+'_'+id2[2]
        else :
            clean_name=img.split('/')[-1].split('.')[0]+'gt.png'

        clean=Image.open(self.clean_dir+clean_name)

        if self.mode=='train':
            i,j,h,w=tfs.RandomCrop.get_params(clean,output_size=(self.size,self.size))
            rain=FF.crop(rain,i,j,h,w)
            clean=FF.crop(clean,i,j,h,w)
            rain,clean=self.augData(rain.convert('RGB'),clean.convert('RGB'))
        else :
            #for test all-img
            rain=tfs.ToTensor()(rain)
            clean=tfs.ToTensor()(clean)
            rain=tfs.Normalize(mean= self.mean,std=self.std)(rain)
        return rain,clean
    def augData(self,data,target):
        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        data=tfs.RandomHorizontalFlip(rand_hor)(data)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data=FF.rotate(data,90*rand_rot)
            target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean= self.mean,std=self.std)(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.rain_img_dirs)

class RainCity_Dataset(data.Dataset):
    def __init__(self,path,size=240,task='rainfog',dataset='RainCityScapes',mode='train'):
        super(RainCity_Dataset,self).__init__()
        self.mean=mean_opt[dataset+task]
        self.std=std_opt[dataset+task]
        self.des=dataset+task
        self.size=size
        self.mode=mode
        self.rain_img_dirs=os.listdir(os.path.join(path,dataset,task,f'{mode}/data'))
        self.rain_imgs=[os.path.join(path,dataset,task,f'{mode}/data/',img) for img in self.rain_img_dirs]
        self.clean_dir=os.path.join(path,dataset,task,f'{mode}/gt/')
    def __getitem__(self, index):
        rain=Image.open(self.rain_imgs[index])
       
        img=self.rain_img_dirs[index]
        
        id=img.split('/')[-1].split('_leftImg8bit')[0]#aachen_00004_000019
        
        clean_name=id+f'_leftImg8bit.png'
        clean=Image.open(self.clean_dir+clean_name)
    
        if rain.size != clean.size :
            print(id,'------------------------------------')
        if self.mode=='train':
            i,j,h,w=tfs.RandomCrop.get_params(clean,output_size=(self.size,self.size))
            rain=FF.crop(rain,i,j,h,w)
            clean=FF.crop(clean,i,j,h,w)
            rain,clean=self.augData(rain.convert('RGB'),clean.convert('RGB'))
        else :
            #for test all-img
            rain=tfs.ToTensor()(rain)
            clean=tfs.ToTensor()(clean)
            rain=tfs.Normalize(mean=self.mean,std=self.std)(rain)
        return rain,clean
    def augData(self,data,target):
        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        data=tfs.RandomHorizontalFlip(rand_hor)(data)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data=FF.rotate(data,90*rand_rot)
            target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=self.mean,std=self.std)(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.rain_img_dirs)



path=os.path.join(cwd,'data')

if opt.trainset=='SPA_REAL' :
    train_loader=DataLoader(dataset=Rain_Dataset_SPAREAL(path,size=opt.crop_size,dataset=opt.trainset,mode='train') , batch_size=opt.bs,shuffle=True)
    test_loader=DataLoader(dataset=Rain_Dataset_SPAREAL(path,size=opt.crop_size,dataset=opt.trainset,mode='test') , batch_size=1,shuffle=False)
elif  opt.trainset=='MPID' or opt.trainset=="Rain100H" or opt.trainset=='Rain100L':
    train_loader=DataLoader(dataset=Rain_Dataset(path,size=opt.crop_size,task=opt.task,dataset=opt.trainset,mode='train') , batch_size=opt.bs,shuffle=True)
    test_loader=DataLoader(dataset=Rain_Dataset(path,size=opt.crop_size,task=opt.task,dataset=opt.trainset,mode='test') , batch_size=1,shuffle=False)
elif opt.trainset=='RainCityScapes' :
    train_loader=DataLoader(dataset=RainCity_Dataset(path,size=opt.crop_size,task=opt.task,dataset=opt.trainset,mode='train') , batch_size=opt.bs,shuffle=True)
    test_loader=DataLoader(dataset=RainCity_Dataset(path,size=opt.crop_size,task=opt.task,dataset=opt.trainset,mode='test') , batch_size=1,shuffle=False)
else :
    pass
if __name__ == "__main__":
    test_loader=DataLoader(dataset=Rain_Dataset(path,size=opt.crop_size,task='mist',dataset='MPID',mode='test') , batch_size=1,shuffle=False)
    for i,(rain,gt) in enumerate(test_loader):
        tensorShow([rain,gt],['rain','gt'])
    # it=iter(test_loader)
    # for i in range(l+10):
    #     rain,gt=next(it)
    #     # tensorShow([rain,gt],['rain','gt'])
    pass

