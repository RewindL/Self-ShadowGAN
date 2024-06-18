import argparse
from matplotlib.image import composite_images
from model import *
from model.GAN_models import *
from model.losses import  TVLoss, MaskTVLoss, OverRangeLoss, CosineSimilarityLoss,MaskStdLoss, MaskAlphaLoss
from utils.image_io import *
from utils.image_processing import *
from utils.common_utils import *
from utils.deep_hist_layers import TripleDimHistLayer, SingleDimHistLayer
from utils.deep_patch_hist_layers import MultiImages_HistLayer
from model.downsampler import *
import time
import shutil
import os 

class ShadowRemoval(object):
    def __init__(self, image_name, output_path, image_torch, mask_torch, iter_num=1600):
        # Base Params
        self.output_path = output_path
        self.iter_num = iter_num
        self.lr_G = 0.0002
        self.lr_D1 = 0.0002
        self.lr_D2 = 0.0002
        self.using_linear_decay = True
        self.eps = 1/255

        # image
        self.image_name = image_name
        self.image_torch = image_torch.cuda()
        self.relit_image_torch = None

        # Histogram-based Disciminator
        self.K = 260            # bins
        self.L = 1/255          # bin_width
        self.W = self.L/2.5     # bandwidth
        self.mu_k = (self.L * (torch.arange(self.K) + 0.5)).view(-1, 1).cuda()
        self.hists_num_per_iter = 32
        self.patches_num_per_hist = 16
        self.patch_w2 = 20
        self.real_hists = None
        self.fake_hists = None

        # Patch-based Disciminator
        self.max_real_patches_num = 200
        self.max_fake_patches_num_in = 100
        self.max_fake_patches_num_bd = 100
        self.patch_w1 = 32
        self.fake_patches = None
        self.real_patches = None
        
        # mask
        self.mask_torch = mask_torch.cuda()
        self.dilated_mask_torch = None
        self.eroded_mask_torch = None
        self.boundary_mask_torch = None
        self.m_in = None
        self.m_out = None

        # net
        self.G_net = None
        self.D1_net = None
        self.D2_net = None
        self.G1_out = None
        self.G2_out = None
        self.G_parameters = None
        self.D1_parameters = None
        self.D2_parameters = None
        
        # init self
        self._init_all()

    def _init_nets(self):
        self.G_net = skip(3, 6,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
        self.D1_net = Discriminator(3).type(torch.cuda.FloatTensor)
        self.D1_net.apply(weights_init_normal)
        self.D2_net = HistDiscriminator_CNN(3, self.K).type(torch.cuda.FloatTensor)
        self.D2_net.apply(weights_init_normal)

    def _init_masks(self):
        e0, d0 = get_eroded_dilated_mask(self.mask_torch, w_in=5, w_out=8)
        e1, d1 = get_eroded_dilated_mask(self.mask_torch, w_in=10, w_out=16)
        self.eroded_mask_torch = e0
        self.dilated_mask_torch = d0
        self.m_in = self.mask_torch - e1
        self.m_out = d1 - self.mask_torch
        self.boundary_mask_torch = d0 - e0

    def _init_parameters(self):
        self.G_parameters = self.G_net.parameters()
        self.D1_parameters = self.D1_net.parameters()
        self.D2_parameters = self.D2_net.parameters()
    
    def _init_optimizers(self):
        self.optimizer_GAN_G = torch.optim.Adam(self.G_parameters, lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_GAN_D1 = torch.optim.Adam(self.D1_parameters, lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_GAN_D2 = torch.optim.Adam(self.D2_parameters, lr=0.0002, betas=(0.5, 0.999))

    def _init_functions(self):
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.smooth_l1_loss = nn.SmoothL1Loss().type(data_type)
        self.mse_loss = nn.MSELoss().type(data_type)
        self.tv_loss = TVLoss().type(data_type)
        self.mask_tv_loss = MaskTVLoss().type(data_type)
        self.sig_bce_loss = torch.nn.BCEWithLogitsLoss().type(data_type)
        self.bce_loss = torch.nn.BCELoss().type(data_type)
        self.overrange_loss = OverRangeLoss().type(data_type)
        self.cs_loss = CosineSimilarityLoss().type(data_type)
        self.mask_alpha_loss = MaskAlphaLoss().type(data_type)
        self.mask_std_loss = MaskStdLoss().type(data_type)
        # histogram functions
        self.single_single_hist = SingleDimHistLayer(self.K, self.L, self.W, self.mu_k) 
        self.single_triple_hist = TripleDimHistLayer(self.K, self.L, self.W, self.mu_k)
        self.multi_hist = MultiImages_HistLayer(self.K, self.L, self.W, self.mu_k)

    def _init_all(self):
        self._init_masks()
        self._init_functions()
        self._init_nets()
        self._init_parameters()
        self._init_optimizers()

    def optimize(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        for j in range(self.iter_num):
            time_flag = time.time()

            self.apply_G_net(j)
            self.update_real_patches()
            self.update_real_hists()            
            self.update_fake_patches()
            self.update_fake_hists()
            
            if(self.using_linear_decay):
                self.linear_decrease_lr("G", start_lr=0.0002, end_lr=0.00002, start_iter=600, end_iter=self.iter_num, iter_now=j)
                self.linear_decrease_lr("D1", start_lr=0.0002, end_lr=0, start_iter=600, end_iter=self.iter_num, iter_now=j)
                self.linear_decrease_lr("D2", start_lr=0.0002, end_lr=0, start_iter=600, end_iter=self.iter_num, iter_now=j)
                
            # D1
            self.optimizer_GAN_D1.zero_grad()
            self.GAN_D1_optimization_closure(j)
            self.optimizer_GAN_D1.step()
            # D2
            self.optimizer_GAN_D2.zero_grad()
            self.GAN_D2_optimization_closure(j)
            self.optimizer_GAN_D2.step()
            # G
            self.optimizer_GAN_G.zero_grad()
            self.G_optimization_closure(j)
            self.optimizer_GAN_G.step()

            iter_time = time.time() - time_flag #iteration time
            print('ImageIndex={} Iter={} iter_time={:.2f}s'.format(self.image_name, j, iter_time), '\r', end='')

    def apply_G_net(self, iteration):
        G_out = self.G_net(self.image_torch)
        ## Apply Relighting Model
        max_relit_rate = 10
        self.COEF_w = (G_out[:,0:3,:,:]*(1-1/max_relit_rate) + 1/max_relit_rate) * self.dilated_mask_torch + (1 - self.dilated_mask_torch)
        self.COEF_b = (10/255) * (G_out[:,3:6,:,:] - 0.5) * self.dilated_mask_torch
        self.relit_image_torch = self.image_torch.detach() / self.COEF_w + self.COEF_b

    def update_real_patches(self):
        self.real_patches = None
        w = self.patch_w1
        kernel_size = (w, w)
        stride = w//6
        patches = get_kernel_patches_out(self.image_torch, self.mask_torch, kernel_size=kernel_size, stride=stride)
        patches = torch.cat((patches, torch.flip(patches, [2]), torch.flip(patches, [3]), torch.flip(patches, [2,3])), dim=0) # Data augmentation
        if(patches.shape[0] > self.max_real_patches_num):
            patches = patches[random.sample(range(patches.shape[0]), self.max_real_patches_num),:,:,:]
        self.real_patches = patches
    
    def update_real_hists(self):
        w = self.patch_w2
        kernel_size = (w, w)
        stride = w//6
        patches =  get_kernel_patches_out_for_hist(self.image_torch, self.mask_torch, kernel_size=kernel_size, stride=stride)
        hists = None   
        for idx in range(self.hists_num_per_iter):
            patches_for_hist = patches[random.sample(range(patches.shape[0]), self.patches_num_per_hist),:,:,:]
            hist = torch.cumsum(torch.mean(self.multi_hist(patches_for_hist), dim=0, keepdim = True), dim=2)
            if(hists == None):
                hists = hist
            else:
                hists = torch.cat((hists, hist), dim=0)
            self.real_hists = hists
    
    def update_fake_patches(self):
        self.fake_patches = None
        w = self.patch_w1
        kernel_size = (w, w)
        stride = w//6
        patches_in = get_kernel_patches_in(self.relit_image_torch, self.mask_torch, kernel_size=kernel_size, stride=stride)
        patches_bd =  get_kernel_patches_bd(self.relit_image_torch, self.boundary_mask_torch, kernel_size=kernel_size, stride=stride)
        if(patches_in.shape[0] > self.max_fake_patches_num_in):
            patches_in = patches_in[random.sample(range(patches_in.shape[0]), self.max_fake_patches_num_in),:,:,:]
        if(patches_bd.shape[0] > self.max_fake_patches_num_bd):
            patches_bd = patches_bd[random.sample(range(patches_bd.shape[0]), self.max_fake_patches_num_bd),:,:,:]
        self.fake_patches = torch.cat((patches_in, patches_bd), dim=0)

    def update_fake_hists(self):
        w = self.patch_w2
        kernel_size = (w, w)
        stride = w//6
        patches = get_kernel_patches_in_for_hist(self.relit_image_torch, self.mask_torch, kernel_size=kernel_size, stride=stride)
        hists = None
        for idx in range(self.hists_num_per_iter):
            patches_for_hist = patches[random.sample(range(patches.shape[0]), self.patches_num_per_hist),:,:,:]
            hist = torch.cumsum(torch.mean(self.multi_hist(patches_for_hist), dim=0, keepdim = True), dim=2)
            if(hists == None):
                hists = hist
            else:
                hists = torch.cat((hists, hist), dim=0)
        self.fake_hists = (hists + self.real_hists)/2

    
    def linear_decrease_lr(self, model, start_lr, end_lr, start_iter, end_iter, iter_now, update_every=1):
        if(iter_now >= start_iter and iter_now <= end_iter and iter_now % update_every == 0):
            k = (start_lr-end_lr)/(start_iter-end_iter) # k < 0
            lr = start_lr + k * (iter_now - start_iter)
            if model == "D1":
                for param_group in self.optimizer_GAN_D1.param_groups:
    	            param_group['lr'] = lr
            if model == "D2":
                for param_group in self.optimizer_GAN_D2.param_groups:
    	            param_group['lr'] = lr
            if model == "G":
                for param_group in self.optimizer_GAN_G.param_groups:
    	            param_group['lr'] = lr
        else:
            pass

    def GAN_D1_optimization_closure(self, iteration):
        self.GAN_D1_loss = 0
        # Reals
        real_loss = 0
        pred = self.D1_net(self.real_patches.detach())
        real_loss += self.sig_bce_loss(pred, self.get_D_real_label(pred.shape).cuda())
        # Fakes
        fake_loss = 0
        pred = self.D1_net(self.fake_patches.detach())
        fake_loss += self.sig_bce_loss(pred, self.get_D_fake_label(pred.shape).cuda())

        self.GAN_D1_loss = 0.02 * (real_loss  + fake_loss)
        self.GAN_D1_loss.backward()

    def GAN_D2_optimization_closure(self, iteration):
        self.GAN_D2_loss = 0
        # Reals
        real_loss = 0
        pred = self.D2_net(self.real_hists.detach())
        real_loss += self.sig_bce_loss(pred, self.get_D_real_label(pred.shape).cuda())
        # Fakes
        fake_loss = 0
        pred = self.D2_net(self.fake_hists.detach())
        fake_loss += self.sig_bce_loss(pred, self.get_D_fake_label(pred.shape).cuda())

        self.GAN_D2_loss = 0.01 * (real_loss + fake_loss)
        self.GAN_D2_loss.backward()

    def G_optimization_closure(self, iteration):
        self.G_Total_loss = 0;self.COEF_loss = 0;self.GAN_G1_loss = 0;self.GAN_G2_loss = 0
        
        self.COEF_loss += (0.8 * self.mask_tv_loss(self.COEF_w, self.eroded_mask_torch) + 0.2 * self.tv_loss(self.COEF_w))
        self.COEF_loss += (0.02 * self.mask_tv_loss(self.COEF_w, self.m_in) + 0.02 * self.mask_tv_loss(self.COEF_w, self.m_out))
        self.COEF_loss += (0.04 * self.mask_tv_loss(self.COEF_b, self.eroded_mask_torch) + 0.01 * self.tv_loss(self.COEF_b))

        pred_patches = self.D1_net(self.fake_patches)
        self.GAN_G1_loss += 0.02 * self.sig_bce_loss(pred_patches, self.get_G_fake_label(pred_patches.shape).cuda())
        pred_hists = self.D2_net(self.fake_hists)
        self.GAN_G2_loss += 0.01 * self.sig_bce_loss(pred_hists, self.get_G_fake_label(pred_hists.shape).cuda())
        self.GAN_G_loss = self.GAN_G1_loss + self.GAN_G2_loss
        self.G_Total_loss += (self.COEF_loss + self.GAN_G_loss)

        self.G_Total_loss += self.overrange_loss(self.relit_image_torch, 0, 1-self.eps, self.dilated_mask_torch)
        self.G_Total_loss += self.cs_loss(self.relit_image_torch, self.image_torch, self.dilated_mask_torch)
        self.G_Total_loss += 0.2 * self.mask_std_loss(self.COEF_w, self.eroded_mask_torch)
        self.G_Total_loss += 0.1 * self.mask_alpha_loss(self.COEF_w, self.m_out.repeat(1,3,1,1), self.m_out, 1)
        
        self.G_Total_loss.backward()

    def save_final_results(self, name):
        save_image("result_{}".format(name), np.clip(torch_to_np(self.relit_image_torch) ,0, 1), output_path=self.output_path)

    def get_D_real_label(self, shape):
        label = torch.ones(shape) - 0.1 * torch.rand(shape)
        return label
    
    def get_D_fake_label(self, shape):
        label = 0.1 * torch.rand(shape)
        return label

    def get_G_fake_label(self, shape):
        label = torch.ones(shape)
        return label

if __name__ == "__main__":
    print("Program begins.")
    time_program_begin = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--idx', type=str, default='1', help='image index')
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(opt.gpu)
    img_name = opt.idx
    print("Parameters set. CUDA={}, image_index={}.".format(opt.gpu, opt.idx))

    img_torch = np_to_torch(prepare_image('images/Shadows/{}.png'.format(img_name)))
    mask_torch = np_to_torch(prepare_image('images/Masks/{}.png'.format(img_name)))
    print("Image and Mask readed.")

    output_path = "output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("[DIR Made]->{}".format(output_path))
    else:
        print("[DIR Exists]->{}".format(output_path))

    s = ShadowRemoval(img_name, output_path, img_torch, mask_torch, iter_num=1600)
    s.optimize()
    s.save_final_results(img_name)
    print("Program ends.")
