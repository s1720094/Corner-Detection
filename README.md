# Corner-Detection
I implemented the Harris Corner Detection algorithm using Python. The steps are shown below:

o Used the formula for the Y-channel of the YIQ model in performing the color-tograyscale image conversion.  
o Computed Ix and Iy correctly by finite differences.  
o Constructed images of Ix2, Iy2, and IxIy correctly.  
o Computed a proper filter size for a Gaussian filter based on its sigma value.  
o Constructed a proper 1D Gaussian filter.  
o Smoothed a 2D image by convolving it with two 1D Gaussian filters.  
o Handled the image border using partial filters in smoothing.  
o Constructed an image of the cornerness function R correctly.  
o Identified potential corners at local maxima in the image of the cornerness function R.  
o Computed the cornerness value and coordinates of the potential corners up to sub-pixel  
accuracy by quadratic approximation.  
o Used the threshold value to identify strong corners for output.  
