import os
import torchvision.models as models
import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image


# Between [0, 1] and float32 type
def read_input(img_path, device='cuda:0'):
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    image = Image.open(img_path)
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = (image - cnn_normalization_mean) / cnn_normalization_std
    return image


def write_output(output_path, output_image):
    print(f"output shape: {output_image.shape}")
    print(f"min: {torch.min(output_image)} max:{torch.max(output_image)}")
    image = transforms.ToPILImage()(output_image)
    image.save(output_path)


def content_loss(model, content, transfered, featured_idx=4):
    conv_index = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
    content_feature_layer = conv_index[featured_idx]
    layers = list(model.children())
    assert(len(layers) > 0)
    conv_layers = layers[0]
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    m_content = content.clone()
    for idx, layer in enumerate(conv_layers):
        if idx == content_feature_layer:
            loss = nn.MSELoss()
            return loss(m_content, transfered)
        m_content = layer(m_content)
        transfered = layer(transfered)


def gram_matrix(features):
    # print(f"feature shape: {features.shape}")
    b,c,w,h = features.shape
    assert b == 1
    # The features.clone() here took me a long time to find out that I need to add.
    # The reason isn't immediately clear to me why this is necessary.
    reshaped_features = torch.reshape(features.clone(), (c, -1))
    # print(f"reshaped feature shape: {reshaped_features.shape}")
    gram_ma = torch.mm(reshaped_features, reshaped_features.t())
    # print(f"gram matrix shape: {gram_ma.shape}")
    return gram_ma


def style_loss(model, style, transfered):
    conv_index = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
    style_feature_layers = set([0, 2, 5, 7, 10])

    layers = list(model.children())
    assert(len(layers) > 0)
    conv_layers = layers[0]

    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()
    losses = []
    m_style = style.clone()

    for idx, layer in enumerate(conv_layers):
        if idx-1 in style_feature_layers:
            losses.append(criterion(gram_matrix(m_content), gram_matrix(transfered)))
            if idx-1 == max(style_feature_layers):
                return sum(losses) * 0.2
        m_content = layer(m_cintent)
        transfered = layer(transfered)


def neural_style_transfer(model, content_image, style_image, num_steps=100, device='cuda:0'):
    transfered_image = torch.zeros(content_image.shape, dtype=content_image.dtype).to(device)
    model = model.to(device)
    run_index = 0
    optimizer = torch.optim.LBFGS(transfered_image.requires_grad_(True))

    while run_index < num_steps:
        def closure():
            optimizer.zero_grad()
            content_l = content_loss(model, content_image, transfered_image)
            style_l = style_loss(model, style_image, transfered_image)
            loss = 0.01 * contnet_l + 100 * style_l
            loss.backward()
            return loss

        run_index += 1
        optimizer.step(closure)

    return transfered_image


if __name__ == "__main__":
  parser = argparse.ArugmentParser()
  parser.add_argument('--content_img_path', type=str, help='input content image file path, source image to perform style transfer on')
  parser.add_argument('--style_img_path', type=str, help="Style image that is used to generate style")
  parser.add_argument('--output_folder', type=str, help='Output folder of the style transfered image, filanems will have _output appended')
  
  device = 'cuda:1'
  vgg19 = models.vgg19(pretrained=True).to(device)
  opt = parser.parse_args()

  content_image = read_input(opt.content_img_path, device)
  style_image = read_input(opt.style_img_path, device)
  output_img = neural_style_transfer(vgg19, content_image, style_image, num_steps=100, device)
  output_img = output_img.cpu().detach()[0]

  output_file_path = os.path.splitext(os.path.basename(content_path))[0]+"_output.png"
  write_output(os.path.join(output_folder, output_file_path), output_img[0])






