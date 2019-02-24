# The image classification CNN served with Flask

The model allows configuring a network architecture by:
- The number of filters at the first convolutional layer.
- The number of convolution blocks composed of single or double convolutional layers with a doubled number of filters with respect to the previous block.
- The number of neurons at the last fully-connected layer.
- The amount of dropout at the fully-connected layer.
- The activation, loss and accuracy functions.

The model allows applying data augmentation provided by `imgaug` module with customizable parameters of horizontal flipping, contrast normalization, and affine transformation. The input images can be pre-processed (by default) to select the central object of interest. The trained model can be served with Docker as a micro-service.

Build a container:
```sh
sudo docker build -t image-cnn:latest .
```

Run traning of a model wit data augmentation and save a model parameters to /app/model-output/params.ini:
```sh
 sudo docker run -v ~/images:/app/images -it -d image-cnn run.py train \
 --images-dir /app/images/*/*.jpeg --network-name net \
 --model-output-dir /app/model-output \
 --model-params-output-path /app/model-output/params.ini \
 --num-epochs 100 --depth 4 --num-base-filters 8 \
 --num-dense-neurons 128 --batch-size 8 --batch-normalization  \
 --learning-rate 0.001 --num-classes 2 --optimizer-name rmsprop \
 --do-augmentation --verbosity 1 --print-model
```

Save state of the container:
```sh
 sudo docker commit DOCKER_CONTAINER_ID image-cnn/trained
```

Run a web-server to perform prediction with a model, which configuration is stored in /app/model-output/params.ini:
```sh
 sudo docker run -p 6000:80 -d -t image-cnn/trained app.py \
 --port 80 --host=0.0.0.0 --model-params-path /app/model-output/params.ini
```

Send a GET request with an ID of shared image at Google Drive:
```sh
curl localhost:6000/predict_gd?image_id=GOOGLE_DRIVE_IMAGE_ID
```
or with URL:
```sh
curl localhost:6000/predict_url?image_url=URL
```
