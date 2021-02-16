# Image classification starter code
Pytorch workspace for classification projects. Few model options available, including a standard CNN Classifier, Pre-trained AlexNet, Pre-trained ResNet50, and other standard architectures.

Code is ready for cpu or gpu training.

Automatic launch of tensorboard (deleting old logs everytime we run train)

First decide classes (Model/models.py)
```
classes = ["class1", "class2", "class3" ...., "class20"]
```

# (Optional) Step 1: Collect Data
This code opens camera stream. Click space bar to take image, and append to labels.csv

For example, if we would like to collect data for label = 2
```
python -m Data.collect -c 2 
```
code will name the images label_#perclass.jpg

# Step 2: Train Model

Select model and loss

For example:
```
model = LastLayer_Alexnet()
```
```
loss = torch.nn.CrossEntropyLoss().to(device)
```
# Step 3: Live inference model

Test your model. 

```
python -m Live_inference.live_inference 
```

if no model given as argument, newest model in Model/saved_models is used
(optional) Prediction default to running majority of a window_size.
