from math import exp
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import math
import numpy as np
from skimage import measure
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
from  torchvision.transforms import ToTensor
import ssim_loss

def ssim(im1,im2):
    im1=im1.clamp(0,1)
    im2=im2.clamp(0,1)
    return ssim_loss.ssim(im1,im2)

def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    d=1.0
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( d / rmse)


if __name__ == "__main__":
    pass
 