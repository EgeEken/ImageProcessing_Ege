# Image Processing tools by Ege

This library includes functions that will help save time while working on image processing projects.
Uses PIP, imageio and numpy

Install steps:

1-) Download and install [Git](https://git-scm.com/downloads)

2-) Run `pip install "git+"https://github.com/EgeEken/ImageProcessing_Ege""` in the terminal

3-) Use `from ImageProcessing_Ege import imgpro` in a python script or runtime (`from ImageProcessing_Ege import imgpro as (prefix)` if you don't want to write the whole thing every time)

Current features:

- Open, load and save functionality to edit and create images using the PIL library
- Open load and save functionality to edit and create videos using the OpenCV library (cv2)
- Functions to turn PIL image objects into numpy matrices for faster processing
- Function to turn images into 1d arrays of normalized black and white versions of them with pixel values between 0 and 1 for neural network training 
- Functions to read and write videos using the opencv library (cv2) and also to create them using tuple RGB matrixes used in and created by the other functions, and vice versa allowing for frames from the videos read by opencv to be treated and processed by PIL and numpy
- Functions to read the current screen and save it as a matrix (primary use case is real time automated reactions to the monitor)


Optimized and improved versions of my image processing programs included: 
- [Simplify](https://github.com/EgeEken/Simplify)
- [Simplify Color V4 and V5](https://github.com/EgeEken/Simplify-Color)
- [Brighten](https://github.com/EgeEken/Brighten), [Saturate](https://github.com/EgeEken/Saturate)
- [Simplify Video](https://github.com/EgeEken/Simplify-Video)
- [Fill Object](https://github.com/EgeEken/Fill-Object)
- [Detect Object](https://github.com/EgeEken/Detect-Object)
