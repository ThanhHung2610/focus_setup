import torch
from PIL import Image
from torchvision import transforms

pretrained_unet_path='./MoFA_UNet_Save/MoFA_UNet_CelebAHQ/unet_200000.model'
device = 'cuda'



unet_for_mask = torch.load(pretrained_unet_path, map_location=device)
print('Loading segmentation network:' + pretrained_unet_path)

#print(unet_for_mask)

image_path = "D:/University/LUAN_VAN/code/FOCUS/input/1.jpg"

org_image = Image.open(image_path)

to_tensor = transforms.ToTensor()

tensor_image = to_tensor(org_image)

dub_tensor = torch.cat((tensor_image,tensor_image),0)

dub_tensor = dub_tensor.reshape(1, dub_tensor.shape[0],  dub_tensor.shape[1],  dub_tensor.shape[2])

print(dub_tensor.shape)

output = unet_for_mask(dub_tensor.to('cuda'))

print(output.shape)