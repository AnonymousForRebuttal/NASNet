from train import *

def test(net,loader_test):
	if os.path.exists(opt.model_dir):
		print(f'resume from {opt.model_dir}')
		ckp=torch.load(opt.model_dir)
		net.load_state_dict(ckp['model'])
		step=ckp['step']
		print(f'---step:{step} ---')
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	l=len(loader_test)
	for i ,(inputs,targets) in enumerate(loader_test):
		print(f'\r eval:{i}/{l}',end='',flush=True)
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		with torch.no_grad():
			pred=net(inputs)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims) ,np.mean(psnrs)

if __name__ == "__main__":
	loader_test=loaders_['test']
	net=models_[opt.net]
	net=net.to(opt.device)
	net=torch.nn.DataParallel(net)
	if opt.device=='cuda':
		cudnn.benchmark=True
	ssim,psnr=test(net,loader_test)
	print(f'ssim : {ssim} | psnr : {psnr}')
	

