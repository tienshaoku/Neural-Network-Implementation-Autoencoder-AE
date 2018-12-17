# Neural-Network-Implementation-Autoencoder-AE


Python code of Neural Network Implementation of an Autoencoder(AE)

1. Description.pdf: description of the homework
2. Review.pdf: review over the homework
3. howtoru.txt, instruction.txt: instruction of running the code
4. num2.py: python code of autoencoder
5. mnist.py: python code to read the data

#### Observations:
1. As this question is meant to compare the training results to the original images, there will be no such thing as accuracy, but only the loss curve.
2. The result of the dimension reduction is similar to that of the previous homework: with larger x value, the image is somewhat more oblique; however, with different y values, there is no explicit distinction.
3. The reconstruction results are somewhat weird, as they do show the shape of original numbers quite clearly, which is a good sign saying that the reconstruction is successful, but they don’t generate the right color of the backgroud. (p.s. the images are gray-scaled)
4. The filters are still the same problem as those of the previous homework: one cannot understand the meaning of these filters.
5. The parametres mentioned above are tried and considered to be leading to better results; however, as many learners of AI concern, I don’t really understand why ends in this conclusion and this set of optimal parametres. I tried to print out some processes of this training, but didn’t yet find out any useful information.
