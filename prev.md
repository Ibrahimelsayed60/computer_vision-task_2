## Adding additive noise to the image

### Uniform Noise:
The noise caused by quantizing the pixels of a sensed image to a number of discrete levels. It has an approximately uniform distribution.

  <img src="Images/uniform_noise.png" style="zoom:50%;" position="Center" />

### Gaussian Noise:
It is a statistical noise having a probability density function equal to normal distribution, also known as Gaussian Distribution. Random Gaussian function is added to Image function to generate this noise.

  <img src="Images/gaussian_noise.png" style="zoom:50%;" position="Center" />

### Salt and Pepper Noise:
An image having salt-and-pepper noise will have a few dark pixels in bright regions and a few bright pixels in dark regions.
  - Randomly pick some pixels in the image to which noise will be added.
  - Color some randomly picked pixels as black setting their value to 0.
  - Color some randomly picked pixels as white setting their value to 255.

<img src="Images/salt_and_pepper_noise.png" style="zoom:50%;" position="Center" />

## Filtering the noisy images
Mask is usually considered to be odded in size so that it has a specific center pixel. This mask is moved on the image such that the center of the mask traverses all image pixels.
### Average Filter:
 It removes the high-frequency content from the image. It is also used to blur an image.

 <img src="Images/average_filter.png" style="zoom:50%;" position="Center" />

### Gaussian Filter:
A Gaussian Filter is a low pass filter used for reducing noise and blurring regions of an image. The filter is implemented as an Odd sized Symmetric Kernel which is passed through each pixel of the Region of Interest to get the desired effect.

  <img src="Images/gaussian_filter.png" style="zoom:50%;" position="Center" />

### Median Filter:
 It is used to eliminate salt and pepper noise. Here the pixel value is replaced by the median value of the neighboring pixel.

 <img src="Images/median_filter.png" style="zoom:50%;" position="Center" />


# Edge Detection

1. Read the image and convert it to grey scale.
2. Calculate rows and columns of original image.
3. Get an empty image which is equal in dimensions to the original image.
4. Apply the edge detection mask to the image using convolution.
5. Display the image after applying edge detection to it by selecting a metthod from the combobox.

## **Sobel Edge Detection**
The matrices associated with Sobel filter :
- Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
- Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

### Result :
<img src="Images\sobel.jpg" style="zoom:85%;" position="Center" />

 ## **Prewitt Edge Detection**
The matrices associated with Prewitt filter :
- Gx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
- Gy = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])

### Result :
<img src="Images\prewitt.jpg" style="zoom:85%;" position="Center" />

## **Roberts Edge Detection**
The matrices associated with Roberts filter :
- Gx = np.array([[1.0, 0.0], [0.0, -1.0]])
- Gy = np.array([[0.0, -1.0], [1.0, 0.0]])

### Result :
<img src="Images\roberts.jpg" style="zoom:85%;" position="Center" />

## Global and Local Threshold

- **Global** thresholding consists of setting an intensity value (threshold) such that all voxels having intensity value below the threshold belong to one phase, the remainder belong to the other.

- Global thresholding is as good as the degree of intensity separation between the two peaks in the image. It is an unsophisticated segmentation choice.

- In global threshold, we put the threshold equal 127. Then, we get the image is larger than the threshold.

  <img src="Images\Global_threshold.png" style="zoom:85%;" position="Center" />

- **Local** thresholding is used to separate desirable foreground image objects from the background based on the difference in pixel intensities of each region.

- Local thresholding selects an individual threshold for each pixel based on the range of intensity values in its local neighborhood. This allows for thresholding of an image whose global intensity histogram doesn't contain distinctive peaks.

  <img src="Images\local_threshold.png" style="zoom:80%;" position="Center"/>

## Transformation

#### Gray Scale Image

- **grayscale** or image is one in which the value of each pixel is a single sample representing only an *amount* of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray. The contrast ranges from black at the weakest intensity to white at the strongest.

- We use the linear equation to get the intensity of the image from RGB channels, Like
  $$
  {\displaystyle Y_{\mathrm {linear} }=0.07R_{\mathrm {linear} }+0.7152G_{\mathrm {linear} }+0.21B_{\mathrm {linear} }}.
  $$

<img src="Images\grayscale_image.png" style="zoom:80%;" position="Center" />

#### Histogram of R,G,B channels

- An **image histogram** is a type of histogram that acts as graphical representation of the tonal distribution in digital image. It plots the number of pixels for each tonal value.

- The horizontal axis of the graph represents the tonal variations, while the vertical axis represents the total number of pixels in that particular tone.

  - Histogram of **Red** channel in image:

    <img src="Images\red_channel_histogram.png" style="zoom:100%;" position="Center"/>

  - Histogram of **Green** channel:

    <img src="Images\green_channel_image.png" style="zoom:100%;" position="Center" />

  - Histogram of **Blue** channel:

    <img src="Images\blue_channel_histogram.png" style="zoom:100%;" position="Center"/>

#### Cumulative Curve

- To draw the **cumulative curve**, we have to compute the cumulative sum of the histogram. The cumulative sum is exactly as it sounds — the sum of all values in the histogram up to that point, taking into account all previous values.

- Then, We now have the cumulative sum, but as you can see, the values are huge (> 6,000,000). We’re going to be matching these values to our original image in the final step, so we have to normalize them to conform to a range of 0–255.

  <img src="Images\cumulative_curve.png" style="zoom:100%;" position="Center" />

## High Pass Filter
1. Read the image and convert it to grey scale.
2. Calculate rows and columns of original image.
3. Get an empty image which is equal in dimensions to the original image.
4. Apply the High pass (Laplacian filter) mask to the image.
5. Perform FFT to the image then inverse fft.
6. Display the image (the result of applying high pass filter to the image is a sharpened image similar to that of edge detection).

### Result :
<img src="Images\HPF.jpg" style="zoom:85%;" position="Center" />

## Low Pass Filter
1. Read the image and convert it to grey scale.
2. Calculate rows and columns of original image.
3. Get an empty image which is equal in dimensions to the original image.
4. Apply the Low pass (Average) mask to the image.
5. Perform FFT to the image then inverse fft.
6. Display the image (the result of applying low pass filter to the image is a blurred image similar to that of Gaussian filter).

### Result :
<img src="Images\lpf.jpg" style="zoom:85%;" position="Center" />

## Hybrid Image

- A **hybrid image** is an image that is perceived in one of two different ways, depending on viewing distance, based on the way humans process visual input.

- In this technique, you have to have two images in the same size and shape, one is applied with **Gaussian** Filter and another is applied with **Laplacian** filter. Then, use filtered images to get new image by using this equation:
  $$
  H = I_1 · \alpha  + I_2 ·(1 − \alpha)
  $$

  - First image:

    <img src="Images\test.jpg" style="zoom:50%;" position="Center" />

  - Second image:

    <img src="Images\test2.jpg" style="zoom:50%;" position="Center"/>

  - **Hybrid** image:

    <img src="Images\hybrid_image.png" style="zoom:70%;" position="Center"/>
