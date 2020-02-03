import torch.nn as nn
import torch

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)
    

class NA(nn.Module):
    def __init__(self,dim_in,dim_out,kernel_size):
        reduction=1
        kernel_size=3
        super(NA,self).__init__()
        self.att=nn.Sequential(
            nn.Conv2d(dim_in,dim_in//reduction,kernel_size,padding=1,groups=dim_in//reduction,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in//reduction,dim_out,1,padding=0,bias=True),
            nn.Sigmoid()
        )
    def forward(self,x):
        out=self.att(x)
        return out

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim*2,dim,kernel_size,bias=True)
        self.attlayer=NA(dim,dim,kernel_size)
    def forward(self, x):
        out1=self.act1(self.conv1(x))
        dense=torch.cat([x,out1],dim=1)
        out2=self.conv2(dense)
        att=self.attlayer(out2)
        out=att*out2
        out += x 
        return out

class Stage(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Stage, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.s = nn.Sequential(*modules)
    def forward(self, x):
        res = self.s(x)
        res += x
        return res

class NASNet(nn.Module):
    def __init__(self,ss,blocks,conv=default_conv):
        super(NASNet, self).__init__()
        self.ss=ss
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        self.stages=nn.ModuleList()
        for _ in range(self.ss):
            self.stages.append(Stage(conv, self.dim, kernel_size,blocks=blocks))
        self.fusion_sub=nn.Sequential(
            conv(self.dim*self.ss,self.dim,kernel_size),
            conv(self.dim,self.dim,1) 
        )
        self.fusion_att=NA(self.dim,self.dim,kernel_size)
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res=[]
        for i in range(self.ss):
            x=self.stages[i](x)
            res.append(x)
        feature=torch.cat(res,dim=1)
        out=self.fusion_sub(feature)
        att=self.fusion_att(out)
        out=att*out
        x=self.post(out)
        return x + x1
if __name__ == "__main__":
    nas=NASNet(1,1)
    print(nas)