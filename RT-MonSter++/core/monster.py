import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *
from core.refinement import REMP
from core.warp import disp_warp
import matplotlib.pyplot as plt
import time
from torchvision import transforms

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
import sys
sys.path.append('./Depth-Anything-V2-list3')
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder

def get_cur_disp_range_samples(cur_disp, ndisp, disp_interval_pixel, shape):
    #shape, (B, H, W)
    #cur_disp: (B, H, W)
    #return disp_range_samples: (B, D, H, W)

    w = float(cur_disp.shape[2])
    # cur_disp_min = (cur_disp - ndisp / 2 * disp_interval_pixel)  # (B, H, W)
    # cur_disp_max = (cur_disp + ndisp / 2 * disp_interval_pixel)
    cur_disp_min = (cur_disp - ndisp / 2 * disp_interval_pixel).clamp(min=0.0)   #(B, H, W)
    cur_disp_max = (cur_disp_min + (ndisp - 1) * disp_interval_pixel).clamp(max=w)

    assert cur_disp.shape == torch.Size(shape), "cur_disp:{}, input shape:{}".format(cur_disp.shape, shape)
    new_interval = (cur_disp_max - cur_disp_min) / (ndisp - 1)  # (B, H, W)

    disp_range_samples = cur_disp_min.unsqueeze(1) + (torch.arange(0, ndisp, device=cur_disp.device,
                                                                    dtype=cur_disp.dtype,
                                                                    requires_grad=False).reshape(1, -1, 1,
                                                                                                1) * new_interval.unsqueeze(1))
    return disp_range_samples  #[B, D, H, W]
            
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift


    
class hourglass_4x(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_4x, self).__init__()

        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                                                padding=1, stride=1, dilation=1)
    def forward(self, x):
        conv = self.conv1(x)

        return conv
    
class hourglass_8x(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_8x, self).__init__()

        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                                                padding=1, stride=1, dilation=1)
    def forward(self, x):

        conv = self.conv1(x)

        return conv


class hourglass_16x(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_16x, self).__init__()

        self.conv1 = BasicConv(in_channels, in_channels, is_3d=True, bn=True, relu=True, kernel_size=3,
                                                padding=1, stride=1, dilation=1)

    def forward(self, x):

        conv = self.conv1(x)

        return conv    
    

class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim[0], kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim[1], kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim[2], kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list



class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)




    def forward(self, features):
        features_mono_list = []

        feat_16x = self.conv16x(features[2]) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)

        return features_mono_list





class Monster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ndisps = [15, 13]  # args.ndisps
        self.disp_interval_pixel = [2, 1]   # args.disp_inteval_pixel
        context_dims = args.hidden_dims[::-1]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.normalize = transforms.Compose([
            transforms.Normalize(mean=mean, std=std) 
        ])

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)

        self.update_block_list = nn.ModuleList([BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps), BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps), BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps)])

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], context_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims)


        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )
        self.spx_16 = nn.ModuleList([
            BasicConv_IN(192, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            ])

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_gru_2x = nn.ModuleList([Conv2x(32, 96, True, keep_concat=False), Conv2x(32, 48, True, keep_concat=False)])
        self.spx_gru_list = nn.ModuleList([nn.Conv2d(96, 9, kernel_size=1, stride=1, padding=0), nn.Conv2d(48, 9, kernel_size=1, stride=1, padding=0)])


        self.conv_list = nn.ModuleList([BasicConv_IN(192+192, 96, kernel_size=3, padding=1, stride=1),
                                        BasicConv_IN(64+96, 96, kernel_size=3, padding=1, stride=1),
                                        BasicConv_IN(48+48, 96, kernel_size=3, padding=1, stride=1)])
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att_list = nn.ModuleList([FeatureAtt(8, 192+192), FeatureAtt(8, 64+96), FeatureAtt(8, 96)])
        self.cost_agg = nn.ModuleList([hourglass_16x(8), hourglass_8x(8), hourglass_4x(8)])
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(f'/home/lwj/MonSter_kitti/checkpoint/depth_anything_v2_{args.encoder}.pth', map_location='cpu')

        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)

        del depth_anything, state_dict_dpt, depth_anything_decoder


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims, ndisps=self.ndisps)



    def infer_mono(self, image1, image2):
        batch_size, _, height_ori, width_ori = image1.shape
        up_size = (height_ori, width_ori)  # H W

        image = torch.cat((image1, image2), 0)
        image_half = (F.interpolate(image, size=(width_ori//2, width_ori//2), mode='bilinear', align_corners=True) / 255.0).contiguous()  
        image_half = self.normalize(image_half)
        resized_image_combined = F.interpolate(image_half, scale_factor=14 / 16, mode='bilinear', align_corners=True)  

        patch_h, patch_w = resized_image_combined.shape[-2] // 14, resized_image_combined.shape[-1] // 14

        features_encoder = self.mono_encoder.get_intermediate_layers(resized_image_combined, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)

        features_left_encoder = tuple([x[:batch_size] for x in lst] for lst in features_encoder)
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        depth_mono = F.interpolate(depth_mono, size=(height_ori//2, width_ori//2), mode='bilinear', align_corners=False)  #   downsample_2x_l.shape[-2:]

        features_4x, features_8x, features_16x, features_32x = self.feat_decoder(features_encoder, patch_h, patch_w)
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = features_4x[:batch_size, :, :, :], features_8x[:batch_size, :, :, :], features_16x[:batch_size, :, :, :], features_32x[:batch_size, :, :, :]
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = features_4x[batch_size:, :, :, :], features_8x[batch_size:, :, :, :], features_16x[batch_size:, :, :, :], features_32x[batch_size:, :, :, :]
        features_left_4x = F.interpolate(features_left_4x, size=tuple(x // 4 for x in up_size), mode='bilinear', align_corners=False)
        features_right_4x = F.interpolate(features_right_4x, size=tuple(x // 4 for x in up_size), mode='bilinear', align_corners=False)
        features_left_8x = F.interpolate(features_left_8x, size=tuple(x // 8 for x in up_size), mode='bilinear', align_corners=False)
        features_right_8x = F.interpolate(features_right_8x, size=tuple(x // 8 for x in up_size), mode='bilinear', align_corners=False)
        features_left_16x = F.interpolate(features_left_16x, size=tuple(x // 16 for x in up_size), mode='bilinear', align_corners=False)
        features_right_16x = F.interpolate(features_right_16x, size=tuple(x // 16 for x in up_size), mode='bilinear', align_corners=False)

        return depth_mono, [features_left_4x, features_left_8x, features_left_16x], [features_right_4x, features_right_8x, features_right_16x] 

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp

    def upsample_disp_x2(self, disp, mask_feat, stem, i):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_gru_2x[i](mask_feat, stem)
        spx_pred = self.spx_gru_list[i](xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample_2x(disp*2., spx_pred).unsqueeze(1)

        return up_disp  #[B, 1, H, W]

    def forward(self, image1, image2, iters=4,  test_mode=False):
        """ Estimate disparity between pair of frames """

        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            depth_mono, features_mono_left,  features_mono_right = self.infer_mono(image1, image2) 

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()


        size = (image1.shape[-2], image1.shape[-1])
        disp_mono_4x = F.interpolate(depth_mono, size=tuple(x // 4 for x in size), mode='bilinear', align_corners=False)
        disp_mono_8x = F.interpolate(depth_mono, size=tuple(x // 8 for x in size), mode='bilinear', align_corners=False)
        disp_mono_16x = F.interpolate(depth_mono, size=tuple(x // 16 for x in size), mode='bilinear', align_corners=False)
        disp_mono_list = [disp_mono_16x, disp_mono_8x, disp_mono_4x]

        features_left = self.feat_transfer(features_mono_left)  # 4x, 8x, 16x, 32x
        features_right = self.feat_transfer(features_mono_right)

        image = torch.cat([image1,image2], 0)
        stem_2x = self.stem_2(image)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2x, stem_2y = torch.chunk(stem_2x, 2, dim=0)
        stem_4x, stem_4y = torch.chunk(stem_4x, 2, dim=0)
        stem_8x, stem_8y = torch.chunk(stem_8x, 2, dim=0)
        stem_16x, stem_16y = torch.chunk(stem_16x, 2, dim=0)


        stem_x_list = [stem_16x, stem_8x, stem_4x, stem_2x]
        stem_y_list = [stem_16y, stem_8y, stem_4y, stem_2y]
        if not test_mode:
            iters_list = [iters//4, iters//4, iters//2]   # 1 1 2
        else:
            iters_list = [1,1,2]  # 1 1 2



        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)

        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [torch.relu(x) for x in inp_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]


        disp_preds = []

        # Coarse-to-fine: 1/16->1/8->1/4
        for i in range(3):

            feat_left = torch.cat((features_left[-1-i], stem_x_list[i]), 1)
            feat_right = torch.cat((features_right[-1-i], stem_y_list[i]), 1)

            match_left = self.desc(self.conv_list[i](feat_left))
            match_right = self.desc(self.conv_list[i](feat_right))


            if i==0:  # 1/16 stage: build full cost volume
                disp_range_samples = None
                gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//16, 8) #[B, 8, D/16, H, W]
                gwc_volume = self.corr_stem(gwc_volume)
                gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
                geo_encoding_volume = self.cost_agg[i](gwc_volume)  

                prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
                init_disp = disparity_regression(prob, self.args.max_disp//16)
                cur_disp = init_disp
                del prob

            else:  # 1/8、1/4 stage: build local cost volume
                disp_range_samples = get_cur_disp_range_samples(cur_disp=disp_next.squeeze(1),
                                                ndisp=self.ndisps[i-1],
                                                disp_interval_pixel=self.disp_interval_pixel[i-1],
                                                shape=[match_left.shape[0], match_left.shape[2], match_left.shape[3]])  #[B, D, H, W]

                gwc_volume = build_gwc_volume_selective(match_left, match_right, disp_range_samples, self.ndisps[i-1], 8)  #[B, 8, ndisps, H, W]
                gwc_volume = self.corr_stem(gwc_volume)
                gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
                geo_encoding_volume = self.cost_agg[i](gwc_volume)  
                cur_disp = disp_next  
            del gwc_volume

            geo_block = Combined_Geo_Encoding_Volume
            geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius[i])
            b, c, h, w = match_left.shape
            coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
            disp = cur_disp


            for itr in range(iters_list[i]):
                disp = disp.detach()

                # lookup
                if i==0:
                    geo_feat = geo_fn(disp, coords)
                else:
                    geo_feat = geo_fn(disp, coords, self.ndisps[i-1])

                # scale and shift depth for multi-scale depth fusion
                bs, _, _, _ = disp.shape
                for b in range(bs):
                    with torch.autocast(device_type='cuda', dtype=torch.float32): 
                        scale, shift = compute_scale_shift(disp_mono_list[i][b].clone().squeeze(1).to(torch.float32), disp[b].clone().squeeze(1).to(torch.float32))
                    disp_mono_list[i][b] = scale * disp_mono_list[i][b] + shift

                if i==2:  # 1/4 stage
                    if itr < int(iters_list[2]-1):  #iters_list[0]
                        net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp, disp_mono_list[i], iter16=(self.args.n_gru_layers==3 and i>=0), iter08=(self.args.n_gru_layers>=2 and i>=1), iter04=(self.args.n_gru_layers>=1 and i>=2))

                    else: # only the last iter of 1/4 resolution conduct dual-branch update
                        warped_right_mono = disp_warp(feat_right, disp_mono_list[i].clone().to(feat_right.dtype))[0]    #feat_right, feat_left
                        flaw_mono = warped_right_mono - feat_left

                        warped_right_stereo = disp_warp(feat_right, disp.clone().to(feat_right.dtype))[0]  
                        flaw_stereo = warped_right_stereo - feat_left 
                        geo_feat_mono = geo_fn(disp_mono_list[i], coords, self.ndisps[i-1])  

                        net_list, mask_feat, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_list[i], geo_feat_mono, iter16=False, iter08=False)
                        net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list, flaw_mono, disp_mono_list[i], geo_feat_mono, flaw_stereo, disp, geo_feat, iter16=False, iter08=False)

                        disp_mono_list[i] = disp_mono_list[i] + delta_disp_mono
                        disp_mono_4x_up = self.upsample_disp(disp_mono_list[i], mask_feat_4_mono, stem_2x)
                        disp_preds.append(disp_mono_4x_up)
                else:  # 1/16、1/8 stage
                    net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp, disp_mono_list[i], iter16=(self.args.n_gru_layers==3 and i>=0), iter08=(self.args.n_gru_layers>=2 and i>=1), iter04=(self.args.n_gru_layers>=1 and i>=2)) 

                disp = disp + delta_disp

                if test_mode and itr < iters_list[2]-1:
                    continue
                if i < 2: 
                    disp_up = F.interpolate(disp * (2**(4-i)), size=size, mode='bilinear', align_corners=True)
                    disp_preds.append(disp_up)
                else:
                    disp_up = self.upsample_disp(disp, mask_feat, stem_2x)
                    disp_preds.append(disp_up)

            disp_next = self.upsample_disp_x2(disp, mask_feat, stem_x_list[i+1], i) if i < 2 else None
            

        if test_mode:
            return disp_up
        
        init_disp_up = F.interpolate(init_disp * 16, size=size, mode='bilinear', align_corners=True)
        
        return init_disp_up, disp_preds, depth_mono