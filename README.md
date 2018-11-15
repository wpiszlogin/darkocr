# darkocr
OCR application for handwritten characters. It uses a dark force of machine learning.

![przyklady3](https://user-images.githubusercontent.com/6407844/48489739-febb7e80-e823-11e8-8786-9324800839fd.png)

## Dependencies
* numpy
* pickle
* matplotlib
* PIL
* tensorflow
* Augmentor

## Solution
The biggest problem was with the unbalanced classes. That is why working on data had the highest priority. 

![histogram](https://user-images.githubusercontent.com/6407844/48527758-8210ba80-e88c-11e8-8126-928bcc16e990.png)

Augmentor library was used for augmentation but the results were unsatisfactory. The characters in the dataset are centered and fit to image boundary, so only some of deformations could be used. However the library allows to extend it. Augmentor was enriched with a few author’s algorithms:
* changing thickness of the line by erosion and dilation
* making variations of aspect ratio only for automatically selected characters - where this has a positive effect  
* autocrop and autocenter image
* smoothing edges of symbols after deformations

![augment_pic](https://user-images.githubusercontent.com/6407844/48491254-69ba8480-e827-11e8-9f41-a8ffd4ccf2ca.png)

The variety of generated images was satisfactory, thus it was decided to setup amount of examples to the largest class set.  
The augmentation can cause overfitting, much attention was given not to mix augmented and original image between training and testing set.  
Another problem was with the duplication of N letter. It was decided to delete single-element class 30 from the training data and not pollute the model. The class was not added to second N-class, because this set is the largest, so it would not impact on the performance. If there is a need to detect single N image, it will be easy to implement additional algorithm, which will compare values of two matrices.  
Some of the classes have small number of examples and the decision about losing them for the validation was hard. That is why k-fold cross validation was used. There were five combinations of subsets for training and testing and five models were created. Final result is made by voting.  
The model is the CNN and consists of three convolution layers and two fully-connected layers. Noticeably improvement in performance and learning speed could be achieved by using batch normalization. It is known that batch normalization has regularization effect, but to make it better drop out layers was added also. Nadam optimizer with learning rate 0.005 was chosen in the experimental way.  
Learning was stopped after 10 epoches. The value was chosen by analyzing the learning graph. It was possible to get a little higher validation accuracy but It was assumed that many training iteration will cause more overfitting in external data set.  

![uczenie](https://user-images.githubusercontent.com/6407844/48528321-a8cff080-e88e-11e8-9d2e-94b94aa26bcb.png)

## Validation
The problem requires predicting one out of 36 characters. There is no information about priority of them, therefore it was assumed they have the same importance.  
Since k-fold method was used with k=5, thus the whole data was used for training and evaluation. Average accuracy of validation data set was __0.9248__. Max was __0.9341__.  
The classification of some symbols (e.g. o and 0) is hard even for a human, therefore it was not expected to achieve accuracy near to 1. Considering this fact, received results are acceptable. It is good to notice, that some classes have from 100 to 200 examples and even perfect augmentation will not reconstruct variation of the handwritten characters. Letters can be written in many different style. The conclusion is that the model will not achieve measured accuracy in real world application. The lack of lower case examples is too big.
Additionally, final model was tested by self-made examples. 140 images were created on a graphics tablet. Average accuracy was 0,8957. By using voting of 5 models (k-fold) the accuracy improved to __0.9286__.  

![validation dark](https://user-images.githubusercontent.com/6407844/48527597-0d3d8080-e88c-11e8-8937-572a89a03cda.png)
