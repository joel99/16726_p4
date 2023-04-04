import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss
from pytorch_lightning import seed_everything

"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# ~12 convs in resnet
def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn).to(device)
    normalization = Normalization().to(device)

    if style_img is not None:
        style_prop = normalization(style_img.to(device))
    else:
        style_layers = []
    if content_img is not None:
        content_prop = normalization(content_img.to(device))
    else:
        content_layers = []

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    class_to_str_map = {
        nn.modules.conv.Conv2d: "conv",
        nn.modules.pooling.MaxPool2d: "pool",
        nn.modules.activation.ReLU: "relu",
    }
    layer_name_ctr = {}
    seq = []

    for i in cnn: # assume vgg 19
        if type(i) == nn.ReLU:
            seq.append(nn.ReLU(inplace=False))
        else:
            seq.append(i)

        if style_img is not None:
            style_prop = seq[-1](style_prop)
        if content_img is not None:
            content_prop = seq[-1](content_prop)

        if type(i) in class_to_str_map:
            label = class_to_str_map[type(i)]
            # 1 index?
            layer_name_ctr[label] = layer_name_ctr.get(label, 0) + 1
            if content_img is not None:
                for content in content_layers:
                    layer_type, layer_ct = content.split('_')
                    if layer_name_ctr[layer_type] == int(layer_ct):
                        content_losses.append(ContentLoss(content_prop))
                        seq.append(content_losses[-1])
            if style_img is not None:
                for style in style_layers:
                    layer_type, layer_ct = style.split('_')
                    if layer_name_ctr[layer_type] == int(layer_ct):
                        style_losses.append(StyleLoss(style_prop))
                        seq.append(style_losses[-1])
        if len(content_losses) == len(content_layers) and len(style_losses) == len(style_layers):
            break
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    model = nn.Sequential(
        normalization,
        *seq,
    )

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, style_layers=style_layers_default, content_layers=content_layers_default, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img, style_layers=style_layers, content_layers=content_layers)
    model.requires_grad_(False)
    input_img.requires_grad_(True)
    # get the optimizer
    optimizer = get_image_optimizer(input_img)
    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)
        # clear the gradients
        optimizer.zero_grad()
        # compute the loss and it's gradient

        model(input_img)
        losses = 0
        if style_losses:
            losses = losses + sum([loss.loss for loss in style_losses]) * style_weight
            # print('style: ', losses)
        if content_losses:
            losses = losses + sum([loss.loss for loss in content_losses]) * content_weight
            # print('content: ', losses)
        losses.backward()
        # return the loss

        return losses

    # run the optimizer
    for i in range(num_steps):
        optimizer.step(closure)
        if i % 50 == 0:
            print("Step {}".format(i))
            # imshow(input_img, title='Input Image')

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img


def main(
    style_img_path,
    style_layers,
    content_img_path,
    content_layers,
    rng=0,
    style_weight=10,
    init_content=False
):
    seed_everything(rng)
    # we've loaded the images for you
    style_img = load_image(style_img_path) if style_img_path else None
    content_img = load_image(content_img_path) if content_img_path else None

    # interative MPL
    plt.ion()

    if style_img is None and content_img is None:
        print("Please specify at least one image to process")
        return

    if style_img is not None and content_img is not None:
        min_x = min(style_img.shape[-2], content_img.shape[-2])
        min_y = min(style_img.shape[-1], content_img.shape[-1])
        style_img = transforms.CenterCrop((min_x, min_y))(style_img)
        content_img = transforms.CenterCrop((min_x, min_y))(content_img)
        if content_img.size(1) == 1:
            content_img = content_img.repeat(1, 3, 1, 1)
        assert style_img.size() == content_img.size(), \
            f"we need to import style and content images of the same size, got {style_img.size()} and {content_img.size()}"

    # plot the original input image:
    # if style_img is not None:
    #     plt.figure()
    #     imshow(style_img, title='Style Image')

    # if content_img is not None:
    #     plt.figure()
    #     imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    if init_content:
        input_img = content_img.clone()
    else:
        input_img = torch.randn(content_img.size() if content_img is not None else style_img.size(), device=device)
    content_layers = content_layers.split(',')
    style_layers = style_layers.split(',')
    output = run_optimization(cnn, content_img, style_img, input_img, style_layers=style_layers, content_layers=content_layers, style_weight=style_weight, num_steps=300)

    plt.figure()
    imshow(output, title='Reconstructed Image')

    # save output to file
    out_label = f"{'init' if init_content else 'rand'}-seed-{rng}"
    if style_img_path:
        out_label += f"-style-{style_img_path.split('/')[-1].split('.')[0]}"
        out_label += f"-style_layers-{style_layers}"
        out_label += f"-style_weight-{style_weight}"
    if content_img_path:
        out_label += f"-content-{content_img_path.split('/')[-1].split('.')[0]}"
        out_label += f"-content_layers-{content_layers}"
    plt.savefig(f'out/{out_label}.png')

    exit(0)
    # texture synthesis
    print("Performing Texture Synthesis from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    # output = synthesize a texture like style_image

    plt.figure()
    imshow(output, title='Synthesized Texture')

    # style transfer
    # input_img = random noise of the size of content_img on the correct device
    # output = transfer the style from the style_img to the content image

    plt.figure()
    imshow(output, title='Output Image from noise')

    print("Performing Style Transfer from content image initialization")
    # input_img = content_img.clone()
    # output = transfer the style from the style_img to the content image

    plt.figure()
    imshow(output, title='Output Image from noise')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('--style_img_path', '-s', type=str, help='path to style image', default="")
    parser.add_argument('--style_layers', '-sl', type=str, help='style layers', default=','.join(style_layers_default))
    parser.add_argument('--content_img_path', '-c', type=str, help='path to content image', default="")
    parser.add_argument('--content_layers', '-cl', type=str, help='content layers', default=','.join(content_layers_default))
    parser.add_argument('--rng', '-r', type=int, help='random seed', default=0)
    parser.add_argument('--style_weight', '-sw', type=float, help='style weight', default=1000000)
    # boolean init-content flag
    parser.add_argument('--init-content', '-ic', action='store_true', help='initialize with content image')

    args = parser.parse_args()
    main(**vars(args))
