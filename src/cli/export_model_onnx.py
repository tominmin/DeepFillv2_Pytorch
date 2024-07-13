import torch
import utils
import os

device = torch.device("cpu")

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

opt = Dict()

opt.gan_type = 'WGAN'
opt.batch_size = 1
opt.in_channels = 4
opt.out_channels = 3
opt.latent_channels = 48
opt.pad_type = 'zero'
opt.activation = 'elu'
opt.norm = 'none'
opt.init_type = 'xavier'
opt.init_gain = 0.02

generator = utils.create_generator(opt).eval()
model_name = "deepfillv2_WGAN_G_epoch40_batchsize4.pth"
model_name = os.path.join(os.path.dirname(__file__),f"../../pretrained_model/{model_name}")


pretrained_dict = torch.load(model_name, map_location=device)
generator.load_state_dict(pretrained_dict)

model = generator
print(model)

dummy_img = torch.randn((1,3,320,320)).cpu()
dummy_mask = torch.randn((1,1,320,320)).cpu()
torch.onnx.export(model, (dummy_img, dummy_mask), f=os.path.join(os.path.dirname(__file__), "../../artifact/deepfillv2_mod.onnx"), verbose=True)