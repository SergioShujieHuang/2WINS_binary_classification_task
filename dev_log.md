First, I need to check the dataset by myself.

It’s like a toyota logo, the background is all black.
According to the document, there are 1000 pictures of good and 350 pictures of bad. it's unbalance.

I think when there are many black, corrupted part or scratched part, it will be labeled as the bad class.

There is no need for data cleaning

PNG size 1024*1024

I try to do data augumentation since the product is a cirle-shaped item.
Also, because there is no significant influence for the RGB image, for process speed, it's better using the gray scale image.

for data augumentation, I am consider to rotate the image.

First, for the RGB image and gray scale image, I am consider to do the experiment.

validation/test set will not be augumented.
explainable AI, Grad-CAM / saliency

initialize anaconda

# RGB and grayscale image experiment with data augumentation
baseline A: ResNet50, RGB, rotation
baseline B: ResNet50, grayscale, roration

resize to 224 * 224

the result shows that 
the RGB image will be influence by the background sometime(red part)

![RGB image](./RGB_grayscale_experiment_with%20data%20augumentation/RBG_influenced_by_bg.png "RBG image")

at the same time, the gray scale image concentrates on the "bad" part

![Grayscale image](./RGB_grayscale_experiment_with%20data%20augumentation/grayscale_bg.png "Grayscale image")

Also there is no significant difference in confusion_matrix and ROC

So based on this experiment, I choose to use grayscale image as input.