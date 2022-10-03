## Virtual Adversarial Training

The idea behind VAT stems from adversarial examples. We can perturb an image with noise so small that it is
invisible to the eye and yet the neural network may produce a different classification for the perturbed image than the original image. 

In VAT we deliberately add noise to an image in such a way that the network misclassifies the image. In adversarial training we add adversarial noise to the images and continue training our network. The network will be robust to test images perturbed with adversarial noise.


In VAT we try to find a perturbation (r) that maximizes
the KL divergence between the original image and the adversarial image. We then train the network to predict the same label for original image and perturbed image. The network is penalized if it predicts a different label for the original and perturbed image.

Set the input arguments accordingly:

```bash
python main.py --dataset cifar10 --num-labeled 4000 --threshold 0.95
```
For example, the above command is to train the model on cifar 10 dataset, using just 4000 labeled images with a threshold of 0.95.