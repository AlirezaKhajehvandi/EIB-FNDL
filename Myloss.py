import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.vgg import vgg16

# adaptive function to use in L_exp
def bright_map_exp(x):
    alpha = 5
    # Scale and shift the sine value to map [-1, 1] to [0, 1]
    y = (1.6 / (torch.exp(alpha * (x - 0.6)) + 1)) - 0.8
    return y

class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):
        b,c,h,w = x.shape
        # x = 1 - x # inverse x
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k
    
#  proposed loss function
class L_color_ratio(nn.Module):
    def __init__(self):
        super(L_color_ratio, self).__init__()

    def forward(self, x , e):

        b,c,h,w = x.shape
        # x = 1 - x # inverse x
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mean_e_rgb = torch.mean(e,[2,3],keepdim=True)
    
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        mr_e,mg_e, mb_e = torch.split(mean_e_rgb, 1, dim=1)

        mrg_ratio = (mr/mg) - (mr_e/mg_e)
        mrb_ratio = (mr/mb) - (mr_e/mb_e)
        mbg_ratio = (mb/mg) - (mb_e/mg_e)

        Drg = torch.pow(mrg_ratio, 2)
        Drb = torch.pow(mrb_ratio, 2)
        Dgb = torch.pow(mbg_ratio, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k

#  proposed loss function
class L_brightness(nn.Module):
    def __init__(self):
        super(L_brightness, self).__init__()

    def forward(self, x ):
        b,c,h,w = x.shape
        # x = 1 - x # inverse x
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        
        # comparative 1/2 this is default so it disappear in name train
        target_brightness_r = mr + ((1 - mr) / 2)
        target_brightness_g = mg + ((1 - mg) / 2)
        target_brightness_b = mb + ((1 - mb) / 2)

        Diff_r = torch.pow((mr - target_brightness_r), 2)
        Diff_g = torch.pow((mg - target_brightness_g), 2)
        Diff_b = torch.pow((mb - target_brightness_b), 2)
        k = torch.pow(Diff_r + Diff_g + Diff_b, 0.5)

        return k

# Original spa loss function
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org , enhance ):
        b,c,h,w = org.shape
        # org = 1 - org ## not inverse if you don't need this line, just clean it
        # enhance = 1 - enhance ## not inverse if you don't need this line, just clean it
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())  # noqa: F841
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)  # noqa: F841

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

#  proposed loss function    
class L_exp(nn.Module):
    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, e, mean_val ):
        # x = 1 - x # inverse x
        b,c,h,w = x.shape

        x = torch.mean(x,1,keepdim=True)
        e = torch.mean(e,1,keepdim=True)
    
        e_mean = self.pool(e)
        x_mean = self.pool(x)
        target_brightness = x_mean + bright_map_exp(x_mean)

        # d = torch.mean(torch.pow(mean- torch.FloatTensor([target_brightness] ).cuda(),2))
        d = torch.mean(torch.pow(e_mean - target_brightness, 2))
        return d



class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        # x = 1 - x # not inverse data thsi is good for presercving detailes
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    




