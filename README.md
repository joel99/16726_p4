# 16726_p4

## Content optimization

Optimizing lower layers (`conv_3`, `conv_4`) preserves content nearly perfectly, but this breaks down phasically at extremely high layers (`conv_7`, `conv_8`). We generate 2 random seeds for both just to be safe.

Layer             |  Image (Seed 0)  | Image (Seed 1)
:---:|:---:|:---:
`conv_3` |  ![phipps_3_0](./out/rand-seed-0-content-phipps-content_layers-%5B'conv_3'%5D.png) | ![phipps_3_1](./out/rand-seed-1-content-phipps-content_layers-%5B'conv_3'%5D.png)
`conv_4` |  ![phipps_4_0](./out/rand-seed-0-content-phipps-content_layers-%5B'conv_4'%5D.png) | ![phipps_4_1](./out/rand-seed-1-content-phipps-content_layers-%5B'conv_4'%5D.png)
`conv_7` |  ![phipps_7_0](./out/rand-seed-0-content-phipps-content_layers-%5B'conv_7'%5D.png) | ![phipps_7_1](./out/rand-seed-1-content-phipps-content_layers-%5B'conv_7'%5D.png)
`conv_8` |  ![phipps_8_0](./out/rand-seed-0-content-phipps-content_layers-%5B'conv_8'%5D.png) | ![phipps_8_1](./out/rand-seed-1-content-phipps-content_layers-%5B'conv_8'%5D.png)

<!-- repeat 3 more times, with conv_4, 7, 8 -->
The effect of noise is small (if it matters at all), and the wide range of consistent reconstruction suggests content layer will not be a critical hyperparameter during joint tuning.

## Style optimization

Style produces quite different but all pleasing results for different layers. We show starry night, 2 seeds, for 4 layers (`conv_2`, `conv_3`, `conv_4`, `conv_5`)

Layer             |  Image (Seed 0)  | Image (Seed 1)
:---:|:---:|:---:
`conv_2` |  ![starry_2_0](./out/rand-seed-0-style-starry_night-style_layers-%5B'conv_2'%5D-style_weight-1000000.png) | ![starry_2_1](./out/rand-seed-1-style-starry_night-style_layers-%5B'conv_2'%5D-style_weight-1000000.png)
`conv_3` |  ![starry_3_0](./out/rand-seed-0-style-starry_night-style_layers-%5B'conv_3'%5D-style_weight-1000000.png) | ![starry_3_1](./out/rand-seed-1-style-starry_night-style_layers-%5B'conv_3'%5D-style_weight-1000000.png)
`conv_4` |  ![starry_4_0](./out/rand-seed-0-style-starry_night-style_layers-%5B'conv_4'%5D-style_weight-1000000.png) | ![starry_4_1](./out/rand-seed-1-style-starry_night-style_layers-%5B'conv_4'%5D-style_weight-1000000.png)
`conv_5` |  ![starry_5_0](./out/rand-seed-0-style-starry_night-style_layers-%5B'conv_5'%5D-style_weight-1000000.png) | ![starry_5_1](./out/rand-seed-1-style-starry_night-style_layers-%5B'conv_5'%5D-style_weight-1000000.png)

The produced style emphasizes much finer arrangements of details at higher layers, which are a bit more consistent with the original style transfer paper. For best results, and also as hinted in code comments, though, we will mix the different layers.

The effect of noise does appear to matter; different colors can dominate. However, the overall textural frequency appears consistent.


# Together
## Implementation
I followed the recommended structure and cached activations in the style/content layers, and used MSE to drive towards these cached activations.
Interestingly, I found noise artifacts if I clipped after each optimization step instead of before. I'm not sure why this occurred.

For tuning, I adjusted the style weight (with the helpful initializations in the comments) by factors of 10 until the style was not overbearing. Thankfully this was only 1 factor of 10.


## Random init 2x2

Content \ Style | <img src="./data/images/style/starry_night.jpeg" width="200px">) | <img src="./data/images/style/picasso.jpg" width="200px">
:---:|:---:|:---:
<img src="./data/images/content/tubingen.jpeg" width="100px"> | ![tubingen_starry](out/rand-seed-0-style-starry_night-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-tubingen-content_layers-['conv_4'].png) | ![tubingen_picasso](out/rand-seed-0-style-picasso-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-tubingen-content_layers-['conv_4'].png)
<img src="./data/images/content/phipps.jpeg" width="100px"> | ![phipps_starry](out/rand-seed-0-style-starry_night-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-phipps-content_layers-['conv_4'].png) | ![phipps_picasso](out/rand-seed-0-style-picasso-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-phipps-content_layers-['conv_4'].png)

If we initialize with the content instead of a random seed:
Content \ Style | <img src="./data/images/style/starry_night.jpeg" width="200px">) | <img src="./data/images/style/picasso.jpg" width="200px">
:---:|:---:|:---:
<img src="./data/images/content/tubingen.jpeg" width="100px"> | ![tubingen_starry](out/init-seed-0-style-starry_night-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-tubingen-content_layers-['conv_4'].png) | ![tubingen_picasso](out/init-seed-0-style-picasso-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-tubingen-content_layers-['conv_4'].png)
<img src="./data/images/content/phipps.jpeg" width="100px"> | ![phipps_starry](out/init-seed-0-style-starry_night-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-phipps-content_layers-['conv_4'].png) | ![phipps_picasso](out/init-seed-0-style-picasso-style_layers-['conv_2',%20'conv_3',%20'conv_4',%20'conv_5']-style_weight-100000.0-content-phipps-content_layers-['conv_4'].png)

The results are pretty subjective, but I would say that Starry Night results are nicer when initialized from content image. Specifically, the textures appear more consistent within each semantic region, rather than everything splashed everywhere. Picasso / Tubingen is similarly improved, but Picasso x Phipps looks the same.

## Personal examples

I adore digital art and took two artists work:
1. [WLOP](https://www.artstation.com/wlop)
![WLOP-work](./data/images/style/wlop.jpg)
1. [Julius Horsthuis](http://www.julius-horsthuis.com/)
![Julius-work](./data/images/style/horsthuis.jpg)
As style sources.

For targets, I tried those with relatively similar content in different domains.
![Cathedral of Learning](./data/images/content/cathy.jpg)
![Gears](./data/images/content/gears.jpg)

Now we transfer:
![cathy_wlop](./out/rand-seed-0-style-wlop-style_layers-%5B'conv_2'%2C%20'conv_3'%2C%20'conv_4'%2C%20'conv_5'%5D-style_weight-100000.0-content-cathy-content_layers-%5B'conv_4'%5D.png)
![gear_horsthuis](./out/rand-seed-0-style-horsthuis-style_layers-%5B'conv_2'%2C%20'conv_3'%2C%20'conv_4'%2C%20'conv_5'%5D-style_weight-100000.0-content-gears-content_layers-%5B'conv_4'%5D.png)

I will say overall, I'm not too happy with these results -- these artworks are so much more than 4 layers of a CNN :).