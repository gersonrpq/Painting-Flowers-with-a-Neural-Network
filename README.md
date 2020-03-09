# Painting Flowers with a Neural Network


<p align="center">
  <img src="Images/test.png" width = 650>
</p>

This project was developed using tensorflow and keras libraries for deep learning.

The main idea of this architecture is to show the power of a Pix2Pix neural network, using a U-net model as generator and a patchGan model as a discriminator.

Painting flower works taking a sketch of a flower to transform it into a real flower. To make the data set, the output images were taken from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html , and input images were made by transforming the outputs using the following function

```
def transforming(address):
    img = cv2.imread(address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.Canny(img, 50, 150)
    img = 255 - img
    return img
```
Notice that the library opencv was used to make the transforming function.

The Neural network architecture can be trained to adapt to any other problem that works similarly to Painting flowers.
On the other hand, the architechture was trained with 100 epochs with 480 random samples taken from the data set. It could be trained a little more.

And this is one of my draws to make the neural network work.

<p align="center">
  <img src="Images/a draw.png" width = 500>
</p>


