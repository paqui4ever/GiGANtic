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
python train.py --model <MODEL_NAME> --epochs <EPOCHS> --batch_size <BATCH_SIZE> --checkpoint_dir <CHECKPOINT_DIR>
```
Available model names are: 
- DCGAN 
- WGAN for Wasserstein GAN with weight clipping
- WGAN with --gradient-penalty flag for Wasserstein GAN with gradient penalty
- InfoGAN
- EBGAN

To resume training from an epoch number N use the --resume_checkpoint flag:

```bash
python train.py --model <MODEL_NAME> --epochs <EPOCHS> --batch_size <BATCH_SIZE> --checkpoint_dir <CHECKPOINT_DIR> --resume_checkpoint <N> 
```

where EPOCHS > N so that the training is done till the total number EPOCHS is reached

## 📈 Results on MNIST dataset

The best performing model was the DCGAN, followed by the InfoGAN and then the WGAN with weight clipping. In the following lines I'll show the final generated images and the evolution of both generator and discriminator losses.

### ♦︎ DCGAN
Trained with 150 epochs and batch size of 1024. The final image and training gif are the following:

<p align="center">
  <img src="assets/DCGAN/DCGANLast.png" alt="DCGAN Final Image" width="256" height="256">
  <img src="assets/DCGAN/DCGAN-MNIST.gif" alt="DCGAN Training GIF" width="256" height="256">
</p>

And the generator loss and discriminator loss plots:
<p align="center">
  <img src="assets/DCGAN/DCGAN_generator_loss.svg" alt="DCGAN Generator Loss" width="300" height="300">
  <img src="assets/DCGAN/DCGAN_discriminator_loss.svg" alt="DCGAN Discriminator Loss" width="300" height="300">
</p>

This model took by far the less number of steps to converge to great quality images compared to the other models.

### ♦︎ WGAN
Trained with 550 epochs and batch size of 1024, and the convolutional layers without the bias term. From epochs 500 to 550 the generated images barely change.

<p align="center">
  <img src="assets/WGAN/WGANLast.png" alt="WGAN Final Image" width="256" height="256">
  <img src="assets/WGAN/WGAN-MNIST.gif" alt="WGAN Training GIF" width="256" height="256">
</p>

The losses for both the generator and discriminator can be seen in the following plots:

<p align="center">
  <img src="assets/WGAN/WGAN_generator_loss.svg" alt="WGAN Generator Loss" width="300" height="300">
  <img src="assets/WGAN/WGAN_discriminator_loss.svg" alt="WGAN Discriminator Loss" width="300" height="300">
</p>

We can see that with a lot greater number of steps, the WGAN with weight clipping with a CNN this small can't achieve the same performance as the DCGAN or InfoGAN despite its loss being very close to 0.

When activating the convolutional layers' bias terms, the model barely changed, with almost the exact same performance. This time I trained it for 500 epochs and the same batch size as before. The final image and training gif are the following:

<p align="center">
  <img src="assets/WGAN-MNISTBias/WGANLastBias.png" alt="WGAN with bias Final Image" width="256" height="256">
  <img src="assets/WGAN-MNISTBias/WGAN-MNISTBias.gif" alt="WGAN with bias Training GIF" width="256" height="256">
</p>

And the losses for both discriminator and generator are:

<p align="center">
  <img src="assets/WGAN-MNISTBias/WGAN_bias_generator_loss.svg" alt="WGAN with bias Generator Loss" width="300" height="300">
  <img src="assets/WGAN-MNISTBias/WGAN_bias_discriminator_loss.svg" alt="WGAN with bias Discriminator Loss" width="300" height="300">
</p>

### ♦︎ WGAN-GP

### ♦︎ InfoGAN
Trained with 200 epochs and batch size of 1024. The final image and training gif are the following:

<p align="center">
  <img src="assets/InfoGAN/InfoGANLast.png" alt="InfoGAN Final Image" width="256" height="256">
  <img src="assets/InfoGAN/InfoGAN-MNIST.gif" alt="InfoGAN Training GIF" width="256" height="256">
</p>

And the generator loss and discriminator loss plots:
<p align="center">
  <img src="assets/InfoGAN/InfoGAN_generator_loss.svg" alt="InfoGAN Generator Loss" width="300" height="300">
  <img src="assets/InfoGAN/InfoGAN_discriminator_loss.svg" alt="InfoGAN Discriminator Loss" width="300" height="300">
</p>

We can see a very different training dynamic compared to the DCGAN, that first learns the background and then the numbers themselves. Meanwhile the InfoGAN learns the numbers and the background at the same time.

It also serves as a great example of how the loss metric in GANs doesn't always correlate with the quality of the generated images. We can see an increasing loss for both the discriminator and the generator on the final part of training, yet the generated images are still improving.

### ♦︎ EBGAN
The convergence time on the EBGAN model was very fast, already outputting good quality images on the first 15 epochs. The final results after 225 epochs of training with a batch size of 1024, are the following:

<p align="center">
  <img src="assets/EBGAN/EBGANLast.png" alt="EBGAN Final Image" width="256" height="256">
  <img src="assets/EBGAN/EBGAN-MNIST.gif" alt="EBGAN Training GIF" width="256" height="256">
</p>

Not much progress could be seen after the first 30 epochs or so, also, there was a slight mode collapse, where no 2, 4 and 5 digits could be found. Furthermore, we can see the loss starting to diverge around that epoch number. Finally, the generator loss and discriminator loss plots:
<p align="center">
  <img src="assets/EBGAN/EBGAN_generator_loss.svg" alt="EBGAN Generator Loss" width="300" height="300">
  <img src="assets/EBGAN/EBGAN_discriminator_loss.svg" alt="EBGAN Discriminator Loss" width="300" height="300">
</p>

## ✒️ Appendix: GAN Objectives 

### 1. GAN Objective

The GAN objective is conceptually defined by a game between two players, the Discriminator and the Generator: we want the Discriminator to learn which images are real and which are fake at the same time that the Generator tries to fool the Discriminator. The formal definition is given by:
$$ \min_G \max_D \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))] $$

### 2. Derivation of the Optimal Discriminator $D^*$

To find the optimal discriminator, we analyze the original GAN value function $V(G,D)$ assuming a fixed generator $G$.

$$
V(G,D) = \int_{\mathcal{X}} p_{data}(x) \log D(x) + p_G(x) \log(1 - D(x)) dx
$$

Since we want to maximize this integral for each point $x$ in the domain $\mathcal{X}$, we focus on maximizing the integrand. That is, we want to maximize a function of the form:

$$
f(y) = a \log y + b \log(1-y)
$$

where we define $y = D(x)$ (our variable to optimize in the range $[0,1]$), $a = p_{data}(x)$, and $b = p_G(x)$.

To find the maximum, we calculate the derivative with respect to $y$ and set it to zero:

$$
f'(y) = \frac{a}{y} - \frac{b}{1-y} = 0
$$

Rearranging the terms:

$$
\frac{a}{y} = \frac{b}{1-y}
$$

Cross-multiplying:

$$
a(1-y) = b y
$$

Expanding the left side:

$$
a - a y = b y
$$

Grouping the terms with $y$:

$$
a = y(a+b)
$$

Solving for $y$:

$$
y = \frac{a}{a+b}
$$

Returning to our original variables, we conclude that the optimal discriminator is:

$$
D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}
$$

---

### 3. Proof of the Jensen-Shannon Divergence

Now that we know the optimal discriminator $D^*(x)$, we replace it in the objective function to see what the generator is actually minimizing:

$$
\mathbb{E}_{x \sim p_{data}} [\log D^*(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D^*(G(z)))]
$$

Replacing the definition of $D^*(x)$:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}\right] + \mathbb{E}_{z \sim p_z} \left[\log \left(1 - \frac{p_{data}(G(z))}{p_{data}(G(z)) + p_G(G(z))}\right)\right]
$$

Simplifying the fraction inside the second logarithm:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}\right] + \mathbb{E}_{z \sim p_z} \left[\log \left(\frac{p_{data}(G(z)) + p_G(G(z)) - p_{data}(G(z))}{p_{data}(G(z)) + p_G(G(z))}\right)\right]
$$

Canceling out the $p_{data}(G(z))$ terms:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}\right] + \mathbb{E}_{z \sim p_z} \left[\log \frac{p_G(G(z))}{p_{data}(G(z)) + p_G(G(z))}\right]
$$

Applying the Change of Variable Theorem (if $z \sim p_z$ and $x = G(z)$, then $x \sim p_G$:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}\right] + \mathbb{E}_{x \sim p_G} \left[\log \frac{p_G(x)}{p_{data}(x) + p_G(x)}\right]
$$

Multiplying and dividing the denominators by 2 to introduce the concept of an average distribution:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{\frac{p_{data}(x) + p_G(x)}{2} \cdot 2}\right] + \mathbb{E}_{x \sim p_G} \left[\log \frac{p_G(x)}{\frac{p_{data}(x) + p_G(x)}{2} \cdot 2}\right]
$$

Applying the logarithm property $\log(\frac{A}{2B}) = \log(\frac{A}{B}) - \log 2$:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{\frac{p_{data}(x) + p_G(x)}{2}}\right] + \mathbb{E}_{x \sim p_G} \left[\log \frac{p_G(x)}{\frac{p_{data}(x) + p_G(x)}{2}}\right] - 2\log 2
$$

Defining the average distribution $M(x) = \frac{p_{data}(x) + p_G(x)}{2}$:

$$
\mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{M(x)}\right] + \mathbb{E}_{x \sim p_G} \left[\log \frac{p_G(x)}{M(x)}\right] - 2\log 2
$$

Recognizing the definition of the Kullback-Leibler Divergence, $D_{KL}(P \parallel Q) = \sum_{\mathcal{X}} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P} \left[\log \frac{P(x)}{Q(x)}\right]$:

$$
D_{KL}(p_{data} \parallel M) + D_{KL}(p_G \parallel M) - 2\log 2
$$

Finally, applying the definition of the Jensen-Shannon Divergence, $D_{JS}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$:

$$
2 D_{JS}(p_{data} \parallel p_G) - 2\log 2
$$

With this, we can see that when the Discriminator is optimal, we are approximating the Jensen-Shannon Divergence between the real and fake distributions. Since the JS Divergence is always non-negative (because it's a symmetric KL divergence with always a finite value because it doesn't depend on the overlap of the two distributions), the minimum value is 0, that is achieved when the two distributions are identical. If we were to reach that state, the Generator would have perfectly learned the data distribution. 

---

### 4. WGAN
The WGAN objective is quite different than the DCGAN one, instead of using the Binary Cross Entropy loss to solve the minimax problem, it uses the Wasserstein distance (also known as the Earth Mover's Distance) to measure the distance between the real and fake distributions. It is given by the following equation:
$$ W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma} [\|x-y\|] = \inf_{\gamma \in \Pi(p_r, p_g)} \int_{\gamma(x, y)} \|x-y\| d\gamma(x, y) $$
where $\gamma$ is the joint distribution of $x$ and $y$ (that gives the probability of $x$ and $y$ occurring together) and $\Pi(p_r, p_g)$ is the set of all joint distributions with marginals $p_r$ and $p_g$. This measure can be thought of as the minimum cost of turning the distribution $p_r$ into $p_g$ by transporting "mass" from one to the other.

In a very high dimensional space we can't compute all the joint distributions, so by the Kantorovich-Rubinstein duality, the metric can be reformulated as:

$$ \min_G \max_D \mathbb{E}_{x \sim p_{data}} [D(x)] - \mathbb{E}_{z \sim p_z} [D(G(z))] $$
where $ D $ is a 1-Lipschitz function, that means that the norm of the gradients of $ D $ are always less than or equal to 1. In order to enforce this constraint we can use the weight clipping method or the gradient penalty method.

I have to add how to do gradient penalty and weight clipping

---

### 5. InfoGAN

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

---

### 6. EBGAN

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


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
      journal="Advances in Neural Information Processing Systems (NeurIPS)",
      year=2017
}
```