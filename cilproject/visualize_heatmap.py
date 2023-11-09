from urllib.request import urlopen
from PIL import Image
import timm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch, cv2

# path = "../data/Train/phase_1/000/000.jpg"
# img = Image.open(path).convert("RGB")
# transform = transforms.Compose([transforms.PILToTensor()])
# img_tensor = transform(img)
# # "repvit_m3.dist_in1k"
# # "resnet152.tv_in1k"
# # "resnet50.tv_in1k"
# # "tf_efficientnetv2_l.in1k"
# model = timm.create_model(
#     "repvit_m3.dist_in1k",
#     pretrained=True,
# )
# model = model.eval()

# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model.forward_features(transforms(img).unsqueeze(0))
# o = torch.squeeze(output[-1])

# head = model.forward_head(output, pre_logits=True)
# c = torch.squeeze(head[-1])
# idx = torch.argmax(c).item()


def return_CAM(feature_maps, weight, class_idx, image_size):
    # nc -> number of channels, h&w -> height and width of feature maps
    nc, h, w = feature_maps.shape
    # flatten all feature map into vectors to perform matrix multiplication
    feature_maps = feature_maps.reshape((nc, h * w))
    print(weight.shape)
    print(class_idx)
    print(feature_maps.shape)
    print(weight[class_idx])
    # calculate CAM by weighting all vectors using corresponding classifier weights
    new_weight = np.asarray([weight[class_idx]] * nc)
    print(new_weight.shape)
    cam = np.matmul(new_weight, feature_maps)
    # reshape back to spatial size of maps
    cam = cam.reshape(h, w)
    # 0-1 normalize
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    # scale to 0-255 pixel value
    cam_img = np.uint8(255 * cam_img)
    print(cam_img.shape)
    # resize to original image size
    cam_img = cv2.resize(cam_img, image_size)
    return cam_img


def gen_heatmap_and_save(img_path, embedding_model, save_path):
    img = Image.open(img_path).convert("RGB")
    # "repvit_m3.dist_in1k"
    # "resnet152.tv_in1k"
    # "resnet50.tv_in1k"
    # "tf_efficientnetv2_l.in1k"
    model = timm.create_model(
        embedding_model,
        pretrained=True,
    )
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = model.forward_features(transforms(img).unsqueeze(0))
    o = torch.squeeze(output[-1])

    head = model.forward_head(output, pre_logits=True)
    c = torch.squeeze(head[-1])

    idx = torch.argmax(c).item()
    img_size = img.size

    cam_img = return_CAM(o.detach().numpy(), c.detach().numpy(), idx, img_size)

    heatmap_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_RAINBOW)
    hybrid_img = cv2.addWeighted(heatmap_img, 0.5, cv2.imread(img_path), 0.5, 0)
    plt.imsave(save_path, hybrid_img)
