# GiGANtic

<p align="center">
    <img src="./assets/banner.png" alt="GiGANtic" width="400">
</p>

GiGANtic is a Generative Adversarial Network library with multiple Pytorch implementations of them. The implemented models are: Wasserstein GAN with weight clipping (WGAN) and gradient penalty (WGAN-GP), Deep Convolutional GAN (DC-GAN), InfoGAN and Energy Based GAN (EBGAN).

## 🚀 Setup 

Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## 🦾 Training

To train a model run:

```bash
python train.py --model <MODEL_NAME> --epochs <EPOCHS> --batch_size <BATCH_SIZE> 
```
Available model names are: 
- DCGAN 
- WGAN for Wasserstein GAN with weight clipping
- WGAN with --gradient-penalty flag for Wasserstein GAN with gradient penalty
- InfoGAN
- EBGAN

To resume training from an epoch number N use the --resume_checkpoint flag:

```bash
python train.py --model <MODEL_NAME> --epochs <EPOCHS> --batch_size <BATCH_SIZE> --resume_checkpoint <N>
```

where EPOCHS > N so that the training is done till the total number EPOCHS is reached

## 📈 Results on MNIST dataset

The best performing model was the DCGAN, followed by the InfoGAN and then the WGAN with weight clipping. In the following lines I'll show the final generated images and the evolution of both generator and discriminator losses.

### DCGAN
Trained with 150 epochs and batch size of 1024. The final image and training gif are the following:

<p align="center">
  <table>
    <tr>
      <td>
        <img src="assets/DCGANLast.png" alt="DCGAN Final Image" width="256" height="256">
      </td>
      <td>
      <img src="assets/DCGAN-MNIST.gif" alt="DCGAN Training GIF" width="256" height="256">
    </td>
  </tr>
</table>
</p>

And the generator loss and discriminator loss plots:
<p align="center">
  <table>
    <tr>
      <td>
        <img src="assets/DCGAN_generator_loss.png" alt="DCGAN Generator Loss" width="256" height="256">
      </td>
      <td>
        <img src="assets/DCGAN_discriminator_loss.png" alt="DCGAN Discriminator Loss" width="256" height="256">
      </td>
    </tr>
  </table>
</p>

This model took by far the less number of steps to converge to great quality images compared to the other models.

### WGAN
Trained with 550 epochs and batch size of 1024, although no visual process can be seen from epochs 500 to 550.

<p align="center">
  <table>
    <tr>
      <td>
        <img src="assets/WGANLast.png" alt="WGAN Final Image" width="256" height="256">
      </td>
      <td>
        <img src="assets/WGAN-MNIST.gif" alt="WGAN Training GIF" width="256" height="256">
      </td>
    </tr>
  </table>
</p>

The losses for both the generator and discriminator can be seen in the following plots:

<p align="center">
  <table>
    <tr>
      <td>
        <img src="assets/WGAN_generator_loss.png" alt="WGAN Generator Loss" width="256" height="256">
      </td>
      <td>
        <img src="assets/WGAN_discriminator_loss.png" alt="WGAN Discriminator Loss" width="256" height="256">
      </td>
    </tr>
  </table>
</p>

We can see that with a lot greater number of steps, the WGAN with weight clipping with a CNN this small can't achieve the same performance as the DCGAN or InfoGAN despite its loss being very close to 0.

### WGAN-GP

### InfoGAN
Trained with 200 epochs and batch size of 1024. The final image and training gif are the following:

<p align="center">
<table>
  <tr>
    <td>
      <img src="assets/InfoGANLast.png" alt="InfoGAN Final Image" width="256" height="256">
    </td>
    <td>
      <img src="assets/InfoGAN-MNIST.gif" alt="InfoGAN Training GIF" width="256" height="256">
    </td>
  </tr>
</table>
</p>

And the generator loss and discriminator loss plots:
<p align="center">
  <table>
    <tr>
      <td>
        <img src="assets/InfoGAN_generator_loss.png" alt="InfoGAN Generator Loss" width="256" height="256">
      </td>
      <td>
        <img src="assets/InfoGAN_discriminator_loss.png" alt="InfoGAN Discriminator Loss" width="256" height="256">
      </td>
    </tr>
  </table>
</p>

We can see a very different training dynamic compared to the DCGAN, that first learns the background and then the numbers themselves. Meanwhile the InfoGAN learns the numbers and the background at the same time.

It also serves as a great example of how the loss metric in GANs doesn't always correlate with the quality of the generated images. We can see an increasing loss for both the discriminator and the generator on the final part of training, yet the generated images are still improving.

### EBGAN


## 📖 References:

```
@article{zhao2017energybased,
      author="Junbo Zhao and Michael Mathieu and Yann LeCun",
      title="Energy-based Generative Adversarial Network",
      journal="International Conference on Learning Representations (ICLR)",
      year=2017
}
```
```
@article{radford2016unsupervised,
      author="Alec Radford and Luke Metz and Soumith Chintala",
      title="Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks",
      journal="International Conference on Learning Representations (ICLR)",
      year=2016
}
```

```
@article{chen2016infogan,
      author="Xi Chen and Yan Duan and Rein Houthooft and John Schulman and Ilya Sutskever and Pieter Abbeel",
      title="InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets",
      journal="Advances in Neural Information Processing Systems (NeurIPS)",
      year=2016
}
```

```
@article{arjovsky2017wasserstein,
      author="Martin Arjovsky and Soumith Chintala and Léon Bottou",
      title="Wasserstein GAN",
      journal="International Conference on Machine Learning (ICML)",
      year=2017
}
```

```
@article{gulrajani2017improved,
      author="Ishaan Gulrajani and Faruk Ahmed and Martin Arjovsky and Vincent Dumoulin and Aaron Courville",
      title="Improved Training of Wasserstein GANs",
      journal="Advances in Neural Information Processing Systems (NIPS)",
      year=2017
}
```