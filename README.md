# Raytracing Renderer
**randering .obj that is built out of triangles with multiple raytracing techniques.**

*All are using Turner Whitted's paper "An Improved Illumination Model for Shaded Display"*

## Featuring multiple techniques to render 3d object with raytracing like:
1) MainRegular.py := The Regular Rigorous way of looping through all things at the same time 
2) MainBVH.py := The BVH way to optimize collision detection
3) MainCuda.py := The multi-threading optimization to optimize ray casting 
4) MainFastCuda.py := The multi-threading optimization to optimize ray casting with BVH addition

**Add requirements.txt to you evn**

## Setup Numba on your computer:
1) Download cuda toolkit 12.6 to your computer
2) Check if it detects CUDA and GPU device using Helpers/TestCuda.py

## More functions:
1) AllAtOnce.py -> Compers all different techniques with output image and time of running
2) Show.py -> Show the 3d object with the ability of the user to move around it freely 

## Setup:
1) Move the .obj into the input folder in this project
2) Change name in the function you run to the name of the .obj file (without the .obj)
3) That's it

## Features I didnt add yet and I want to:
1) .obj color
2) lights color(it doesnt work as i want it to wark)
3) .obj that isnt built out of triangles(ball, curves and more 3d shapes)
4) basic ui for all this project

