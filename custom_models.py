import torch
import torch.nn as nn
import timm
import clip


DEVICE = 'cuda'
CLIP_BACKBONE = 'RN50x4'
USE_IVE = False
USE_DVE = False
USE_DENSENET = True
USE_ICVE = False
COARSE_CLASS_NUM = 14
IVE_MODEL_TYPE = 'convnext_small'
FINE_CLASS_NUM = 410
IVE_FINE_PATH = '/home/otabek.nazarov/Downloads/thesis/chest-xray-project/saved/convnext_fine_best_model_weights.pt'
DVE_PATH = '/home/otabek.nazarov/Downloads/thesis/x-ray-report-generation/saved/clip_pretrain_best.pth.tar'
IVE_COARSE_PATH = '/home/otabek.nazarov/Downloads/thesis/chest-xray-project/saved/convnext_coarse_best_model_weights.pt'
ICVE_PATH = '/home/otabek.nazarov/Downloads/thesis/x-ray-report-generation/saved/clip_cluster_pretrain_best.pth.tar'


class CustomClip(nn.Module):
    def __init__(self):
        super(CustomClip, self).__init__()
        device = torch.device(DEVICE)
        self.clip_model, preprocess = clip.load(CLIP_BACKBONE, device=device, jit=False)
        self.visual = self.clip_model.visual
                
    def forward(self, images, tokens):
        return self.clip_model(images, tokens)


class CustomEncoder(nn.Module):
    def __init__(self):
        super(CustomEncoder, self).__init__()

        device = torch.device(DEVICE)

        # Load IVE model if required
        if USE_IVE:
            self.feature_dim = 768
            # Load coarse model
            self.coarse_ive = timm.create_model(IVE_MODEL_TYPE, 
                                                pretrained=False, 
                                                num_classes=COARSE_CLASS_NUM)
            coarse_model_weights = torch.load(IVE_COARSE_PATH, map_location=device)
            coarse_model_weights = rename_state_dict_keys(coarse_model_weights)
            self.coarse_ive.load_state_dict(coarse_model_weights)
            self.coarse_ive.reset_classifier(0)
            freeze_layers(self.coarse_ive)

            # Load fine-grained model
            self.fine_ive = timm.create_model(IVE_MODEL_TYPE, 
                                              pretrained=False, 
                                              num_classes=FINE_CLASS_NUM)
            fine_model_weights = torch.load(IVE_FINE_PATH, map_location=device)
            fine_model_weights = rename_state_dict_keys(fine_model_weights)
            self.fine_ive.load_state_dict(fine_model_weights)
            self.fine_ive.reset_classifier(0)
            freeze_layers(self.fine_ive)

        # Load DVE model if required
        if USE_DVE:
            clip_model = CustomClip()
            checkpoint = torch.load(DVE_PATH)
            load_checkpoint(checkpoint, clip_model)
            self.dve = clip_model.visual
            self.dve.attnpool = nn.Identity()
            freeze_layers(self.dve)
            self.feature_dim = 2560

        if USE_DENSENET:
            self.densenet = timm.create_model('densenet121', 
                                              pretrained=False)
            self.feature_dim = 1024

        # Load ICVE model if required
        if USE_ICVE: 
            self.icve = XRayClusterModel()
            checkpoint = torch.load(ICVE_PATH)
            load_checkpoint(checkpoint, self.icve)
            freeze_layers(self.icve)
            self.feature_dim = 640


    def forward(self, images):
        visual_features = []
        # Get DVE features
        if USE_DVE:
            dve_features = self.dve(images)
            dve_features = dve_features.float()
            visual_features.append(dve_features)

        # Get IVE features
        if USE_IVE:
            coarse_ive_features = self.coarse_ive.forward_features(images).flatten(start_dim=-2, end_dim=-1)
            fine_ive_features = self.fine_ive.forward_features(images).flatten(start_dim=-2, end_dim=-1)
            visual_features.append(coarse_ive_features)
            visual_features.append(fine_ive_features)

        if USE_DENSENET:
            densenet_features = self.densenet.forward_features(images)
            visual_features.append(densenet_features)

        if USE_ICVE:
            icve_features = self.icve.extract_all_fast(images)
            icve_features = icve_features.float()
            icve_features = torch.permute(icve_features, (0, 2, 1))
            visual_features.append(icve_features)
        
        if USE_IVE or USE_ICVE:
            visual_features = torch.cat(visual_features, dim=2)
        else:
            visual_features = torch.cat(visual_features)
            visual_features = visual_features.reshape(images.shape[0], -1, visual_features.shape[2], visual_features.shape[3])

        return visual_features



def freeze_layers(model):
    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False


def rename_state_dict_keys(state_dict, str_to_remove='model.'):
    for key in list(state_dict.keys()):
        state_dict[key.replace(str_to_remove, '')] = state_dict.pop(key)
    return state_dict


def load_checkpoint(checkpoint, model, device='cuda:0', optimizer=None, multi_gpu=False):
    print("=> Loading checkpoint")
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



class XRayClusterModel(nn.Module):
    def __init__(self):
        super(XRayClusterModel, self).__init__()
        self.device = torch.device(DEVICE)
        self.num_clusters = 13
        dims=[16, 3*self.num_clusters]

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=5, stride=1, padding=2),
            LayerNorm(dims[0], eps=1e-6),
            nn.GELU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[1], eps=1e-6),
            nn.GELU()
        )

        self.clip_model = CustomClip()
        self.clip_vis = self.clip_model.visual            


    def extract_all(self, images):
        # Run convolution to get K cluster channels
        convolved_images = self.layer0(images)
        convolved_images = self.layer1(convolved_images)

        # Get Kth clusters features
        embeddings_list = []
        
        for i in range(self.num_clusters):
            # Reshape to calculate only for Kth cluster and then pass it to CLIP
            embeddings = self.clip_vis(convolved_images[:,3*i:3*i+3,:,:])
            embeddings_list.append(embeddings.unsqueeze(1).detach())
            
        stacked_embeddings = torch.cat(embeddings_list, dim=1)
        
        return stacked_embeddings

    def extract_all_fast(self, images):
        # Run convolution to get K cluster channels
        convolved_images = self.layer0(images)
        convolved_images = self.layer1(convolved_images)

        # Reshape for clip 
        conv_shape = convolved_images.shape
        reshaped_conv_images = convolved_images.view(-1, 3, conv_shape[-2], conv_shape[-1])

        # Get K clusters embeddings
        embeddings = self.clip_vis(reshaped_conv_images)
        embeddings = embeddings.view(-1, self.num_clusters, embeddings.shape[-1])
        
        return embeddings

    
    def get_convolved_images(self, images):
        # Run convolution to get K cluster channels
        convolved_images = self.layer0(images)
        convolved_images = self.layer1(convolved_images)
        return convolved_images

    
    def extract_label_embeddings(self, images, labels):
        # Run convolution to get K cluster channels
        convolved_images = self.layer0(images)
        convolved_images = self.layer1(convolved_images)
        
        # Select channel based on current cluster
        selected_channel_images = []
        # selected_channel_images = torch.nn.ModuleList()
        for idx, k_label in enumerate(labels):
            # Reshape to calculate only for Kth cluster
            c_idx = 3 * k_label.item()
            selected_channel_images.append(convolved_images[idx,c_idx:c_idx+3,:,:].unsqueeze(0))
        
        # Reshape for CLIP
        clip_images = torch.cat(selected_channel_images)

        # Arcface output
        embeddings = self.clip_vis(clip_images)

        return embeddings

    
    def forward(self, images, texts, labels):
        # Run convolution to get K cluster channels
        convolved_images = self.layer0(images)
        convolved_images = self.layer1(convolved_images)
        
        # Select channel based on current cluster
        selected_channel_images = []
        # selected_channel_images = torch.nn.ModuleList()
        for idx, k_label in enumerate(labels):
            # Reshape to calculate only for Kth cluster
            c_idx = 3 * k_label.item()
            selected_channel_images.append(convolved_images[idx,c_idx:c_idx+3,:,:].unsqueeze(0))
        
        # Reshape for CLIP
        clip_images = torch.cat(selected_channel_images)

        # CLIP output
        logits_per_image, logits_per_text = self.clip_model(clip_images, texts)

        return logits_per_image, logits_per_text




if __name__ == "__main__":
    device = torch.device("cuda")

    img_x = torch.rand(2, 3, 294, 294).to(device)
    
    # out = model(img_x, tokens, labels)
    # out = model.extract_all(img_x)
    # out = model.extract_label_embeddings(img_x, labels)
    # print(out.shape)
    model = CustomEncoder()
    model.to(device)
    out = model(img_x)
    print(out.shape)