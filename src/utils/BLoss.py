'''
Author: fangzuliang
Create Date: 2020-07-30
Modify Date: 2020-07-30
descirption: ""
Editted to meet quantitative precipitation estimation needs
'''
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class MELoss(torch.nn.Module):
    '''
    define the mean error (not mean absolute error). No matter the prediction is smaller than ground-truth
     or bigger than the ground truth, the func just calculate the bias of every sample and then average them.
    '''
    def __init__(self):
        super(MELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(y_pred - y_true)


class BMELoss(torch.nn.Module):
    '''
    define the weighted mean error (not mean absolute error). No matter the prediction is smaller than ground-truth
     or bigger than the ground truth, the func just calculate the bias of every sample and then average them.
    '''
    def __init__(self, weights = [0.3, 1, 1.2, 1.6, 2, 10], thresholds = [1,2,4,8,20,50]):
        super(BMELoss, self).__init__()

        assert len(weights) == len(thresholds)
        self.weights = weights
        self.thresholds = thresholds

    def forward(self, y_pre, y_true):
        w = torch.zeros_like(y_true)
        n = len(self.weights)
        for i in range(n):
            if i==0:
                mini = 0
            else:
                mini = self.thresholds[i-1]

            mask = (y_true>=mini) * (y_true<self.thresholds[i])
            w[mask] = self.weights[i]

        w[y_true>=self.thresholds[-1]] = self.weights[-1]

        return torch.mean(w * (y_pre - y_true))


class BMAELoss(torch.nn.Module):
    '''
    func: MAE损失中给强回波处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    '''
    def __init__(self, weights = [0.3, 1, 1.2, 1.6, 2, 10], thresholds = [1, 2, 4, 8, 20, 50]):
        super(BMAELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        # scale = max(thresholds)
        self.weights = weights
        # self.thresholds =[threshold/scale for threshold in thresholds] 
        self.thresholds =thresholds
        
    def forward(self,y_pre,y_true):
        w = torch.zeros_like(y_true)
        n = len(self.weights)
        for i in range(n):
            if i==0:
                mini = 0
            else:
                mini = self.thresholds[i-1]
                
            mask = (y_true>=mini) * (y_true<self.thresholds[i])
            w[mask] = self.weights[i]
            
        w[y_true>=self.thresholds[-1]] = self.weights[-1]
            
        return torch.mean(w * (abs(y_pre - y_true)))
    

class BMSELoss(torch.nn.Module):
    '''
    func: MSE损失中给强回波处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    '''
    def __init__(self, weights = [20,30,40,50,80], thresholds = [15, 25, 35, 45, 55]):
        super(BMSELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        # scale = max(thresholds)
        self.weights = weights
        # self.thresholds = [threshold/scale for threshold in thresholds] 
        self.thresholds = thresholds
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
    def forward(self, y_pre, y_true, weight=None):
        w = torch.zeros_like(y_true)
        n = len(self.weights)
        for i in range(n):
            if i==0:
                mini = 0
            else:
                mini = self.thresholds[i-1]
                
            mask = (y_true>=mini) * (y_true<self.thresholds[i])
            w[mask] = self.weights[i]
        
        w[y_true>=self.thresholds[-1]] = self.weights[-1]
        
        if weight is not None:
            return torch.sum(w * (y_pre - y_true)**2)/weight.sum()
        else:
            return torch.mean(w * (y_pre - y_true)**2)
    

class BMSAELoss(torch.nn.Module):
    '''
    func: MSE和MAE损失中给强回波处的误差更大的权重，同时将BMSE 和 BMAE按照不同权重累加起来
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    mse_w: float
        mse权重, default 1
    mae_w: float
        mae权重, default 1
    '''
    def __init__(self, weights = [1.5, 2.5, 3.75, 5, 6.25, 10], 
                 thresholds = [1,2,4,8,20,50],
                 mse_w = 1,mae_w = 1):
        super(BMSAELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        # scale = max(thresholds)
        self.weights = weights
        # self.thresholds = [threshold/scale for threshold in thresholds] 
        self.thresholds = thresholds
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        self.mse_w = mse_w
        self.mae_w = mae_w
        
    def forward(self,y_pre,y_true):
        w = torch.zeros_like(y_true)
        n = len(self.weights)
        for i in range(n):
            if i==0:
                mini = 0
            else:
                mini = self.thresholds[i-1]
                
            mask = (y_true>=mini) * (y_true<self.thresholds[i])
            w[mask] = self.weights[i]
            
        w[y_true>=self.thresholds[-1]] = self.weights[-1]
        #return self.mse_w*torch.mean(w * (y_pre - y_true)**2) + self.mae_w*torch.mean(w * (abs(y_pre - y_true)))
        return self.mse_w*torch.sum(w * (y_pre - y_true)**2) / w.sum() + self.mae_w*torch.sum(w * (abs(y_pre - y_true))) / w.sum()
    

class STBMSELoss(torch.nn.Module):
    '''
    func: MSE损失在空间中给强回波处的误差更大的权重,在时间序列上给时间靠后帧更多的权重  
    Parameter
    ---------
    spatial_weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    time_weight_gap: int
        给不同时间雷达帧损失不同的权重，默认weight(t+1) - weight(t) = time_weight_gap
        default 1.如果为0，则表示所有时间帧权重一致
    '''
    def __init__(self, spatial_weights = [1,2,5,10,30],
                 thresholds = [20,30,40,50,80],time_weight_gap = 0.5):
        super(STBMSELoss,self).__init__()
        
        assert len(spatial_weights) == len(thresholds)
        scale = max(thresholds)
        self.spatial_weights = spatial_weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
        self.time_weight_gap = time_weight_gap
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        '''
        w_true = y_true.clone()
        for i in range(len(self.spatial_weights)):
            w_true[w_true < self.thresholds[i]] = self.spatial_weights[i] #获取权重矩阵
        
        
        if len(y_true.size()) == 4:
            batch, seq, height, width = y_true.shape
            # y_true = np.expand_dims(y_true,axis = 2)
            # y_pre = np.expand_dims(y_pre,axis = 2)
            # w_true = np.expand_dims(w_true,axis = 2)
        
        if len(y_true.size()) == 5:
            batch,seq, channel,height,width = y_true.shape
            assert channel == 1
            
        time_weight = torch.arange(0,seq)*self.time_weight_gap + 1 
        
        all_loss = 0
        for i in range(seq):
            loss = torch.mean(w_true[:,i]*(y_pre[:,i] - y_true[:,i])**2)
            all_loss += time_weight[i]*loss
        
        return all_loss


class CBLoss(torch.nn.Module):
    def __init__(self, beta, thresholds):
        super(CBLoss, self).__init__()
        self.beta = beta
        self.thresholds = thresholds

    def forward(self, y_true, y_pred):
        w = torch.zeros_like(y_true)
        n = len(self.thresholds)
        for i in range(n):
            if i==0:
                mini = 0
            else:
                mini = self.thresholds[i-1]

            mask = (y_true>=mini) * (y_true<self.thresholds[i])
            w[mask] = (1 - self.beta) / (1 - torch.pow(self.beta, mask.sum()))

        #print(w.max(), w.min())
        w[y_true>=self.thresholds[-1]] = (1 - self.beta) / ( 1 - torch.pow(self.beta, (y_true>=self.thresholds[-1]).sum()))
        return torch.mean(w * (y_true-y_pred)**2)*1e8


class SSIMLoss(torch.nn.Module):
    def __init__(self, C1=0.01, C2=0.04, C3=0.04/2):
        super(SSIMLoss, self).__init__()
        self.c1 = C1
        self.c2 = C2
        self.c3 = C3

    def forward(self, y_pred, y_true):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        mu_pred = torch.mean(y_pred)
        mu_true = torch.mean(y_true)
        lmu = (2*mu_pred*mu_true + self.c1) / (torch.pow(mu_pred, 2) + torch.pow(mu_true, 2) + self.c1)

        sigma_pred = torch.std(y_pred)
        sigma_true = torch.std(y_true)
        lsigma = (2*sigma_pred*sigma_true + self.c2) / (torch.pow(sigma_pred, 2) + torch.pow(sigma_true, 2) + self.c2)

        cosigma = 1 / (len(y_pred)-1) * torch.sum((y_pred-mu_pred) * (y_true-mu_true))
        lstrue = (cosigma + self.c3) / (sigma_pred*sigma_true + self.c3)
        # return (lmu*lsigma*lstrue)
        mse = torch.mean((y_pred - y_true)**2)
        ssim = lmu*lsigma*lstrue
        return mse, ssim, 0.1*mse + 0.1*(1-ssim)*25


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class LatWeightedMSE(torch.nn.Module):
    def __init__(self, lat_weights, device):
        self.lat_weights = torch.tensor(lat_weights).to(device)
        super(LatWeightedMSE, self).__init__()
        
    def forward(self, y_true, y_pred):
        loss = (y_true-y_pred)**2 * self.lat_weights[None, None]
        loss = torch.mean(loss)
        return loss


class MixedLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = torch.nn.BCELoss()  # Binary Cross Entropy Loss
        self.mse_loss = torch.nn.MSELoss(reduction='none')  # Mean Squared Error Loss

    def forward(self, y_true, y_pred):
        # Binary Cross Entropy Loss
        y_pred = y_pred.flatten()
        y_true = y_true.flatten() 
        y_pred_mask = (y_pred > 0) * 1.0
        y_true_mask = (y_true > 0) * 1.0
        region_loss = self.bce_loss(y_pred_mask, y_true_mask)

        # intenstity loss in precipitation region
        intensity_loss = self.mse_loss(y_pred, y_true)
        intensity_loss = torch.sum(intensity_loss * y_true_mask) / torch.sum(y_true_mask)

        # total loss
        total_loss = self.alpha * region_loss + self.beta * intensity_loss
        return total_loss


if __name__ == "__main__":
    import torch

    a = torch.rand((2, 1, 10, 10))
    b = torch.rand((2, 1, 10, 10))

    # loss_fn = STBMSELoss()
    # loss_fn = BMSELoss()
    # loss_fn = BMAELoss()
    # loss_fn = BMSAELoss()
    # loss_fn = MSE_IOULoss()
    # loss_fn = SSIMLoss()
    loss_fn = MixedLoss()
    loss = loss_fn(a, b)

    print(loss)
    

    