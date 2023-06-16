<!--
###############################################################################################################################################################
＃　this file is to decribe the data pre-processing process and Model select in image_classification_nn.py
###############################################################################################################################################################
function in used:

function FeatureExtractorGrayscale is used to load the images and covert to grayscale. 

function top1_5_score is used to calculate the TOP-1, TOP-5 Accuracy from the prediction of each model.

function colorQuant is used to do the quantization through K-Means. Cluster number is set to be 2 in this case
###############################################################################################################################################################
model selection:

model can be directly called from the following code, with loss function 'categorical_crossentropy'
by default, the input image sizes = 64*64*1 batch size = 128, epoch = 10

 --> 