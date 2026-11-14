# Maths Image Lab | Convolution & Inner Product

A GUI image editor built with Tkinter, OpenCV, NumPy, and Pillow. This application allows users to explore mathematical image processing concepts, such as convolution and local inner products, through an interactive graphical interface.

## Features
* **Image I/O:** Open and save images in formats including PNG, JPG, BMP, and TIFF.
* **Grayscale Conversion:** Transform color BGR images into single-channel intensity values.
* **Blurring Techniques:** Apply a 3x3 Averaging Blur (equal contribution) and a 5x5 Gaussian Blur weighted kernel.
* **Sharpening:** Emphasize local details by utilizing a convolution kernel that subtracts surrounding pixels from the emphasized center.
* **Edge Detection:** Apply a Laplacian kernel to highlight rapid intensity changes, and utilize Sobel (X) kernels to map vertical edges.
* **Inner Product Response:** Treat a local 3x3 image patch as a vector of 9 values and compute its dot product with a 3x3 filter kernel to map the local response at each pixel.
* **Live Previews:** Side-by-side graphical comparison of the original loaded image and the processed output.

## Prerequisites
Ensure you have Python installed along with the required dependencies. 

Install the required packages using pip:
```bash
pip install opencv-python numpy Pillow