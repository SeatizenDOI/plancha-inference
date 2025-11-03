This workflow is model agnostic to perform inference on Seatizen session.

We used a pre-model named Jacques to filter useless images.

We can plug any model we want by adding a class with the good format :

- the class need to specify the tensorrt implementation if exists
- take in case the saver method.
- if she want the predictions associated with GPS data
- if she want the prediction generate as a raster
- If she want to add something to the pdf preview
