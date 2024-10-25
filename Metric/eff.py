from thop import profile
import torch
from torchvision.models import resnet50
import time
import model

def cal_eff_score(count = 100, use_cuda=True):

    # define input tensor
    # inp_tensor = torch.rand(1, 3, 1080, 1080) # NOTE: this is the shape for ACDC images
    inp_tensor = torch.rand(1, 3, 720, 1280) # NOTE: this is the shape for ACDC images


    model_ = model.enhance_net_nopool(1).cuda()
    model_.load_state_dict(torch.load('./model_parameter/final/Epoch99.pth'))

    # define model_
    # model_ = resnet50()

    # deploy to cuda
    if use_cuda:
        inp_tensor = inp_tensor.cuda()
        model_ = model_.cuda()

    # get flops and params
    flops, params = profile(model_, inputs=(inp_tensor, ))
    G_flops = flops * 1e-9
    M_params = params * 1e-6

    # get time
    start_time = time.time()
    for i in range(count):
        _ = model_(inp_tensor)
    used_time = time.time() - start_time
    ave_time = used_time / count

    # print score
    print('FLOPs (G) = {:.4f}'.format(G_flops))
    print('Params (M) = {:.4f}'.format(M_params))
    print('Time (S) = {:.4f}'.format(ave_time))

if __name__ == "__main__":
    cal_eff_score()