we learnt a lot of new things 

first we did rgb to grayscale conversion using the luminance formula 

gray[index] = 0.299*R + 0.587*G + 0.114*B 

we also saw how to calculate indexes for r,g,b part of a pixel 


then we did grayscale to blur conversion 

in this we take sum of all the neighbors in the blur_radius and then take average to set it as the new value

then in the last kernel we convolve our matrix with a filter 

the centre pixel value is set as the sum of all the products and then taken average with the kernel sum


refer this link https://youtu.be/C_zFhWdM4ic?si=nLzxQu5o-k3esM6i 