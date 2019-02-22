# image-classification-cnn
The image classification CNN served with Flask

Build a container.
```sh
sudo docker build -t image-cnn:latest .
```

Run traning of a model with a default parameters and save a model parameters to /app/model-output/params.ini.
```sh
sudo docker run -v ~/images:/app/images -p 5000:80 -t -i image-cnn run.py \
train --images-dir /app/images/*/*.jpeg --network-name net \
--model-output-dir /app/model-output --model-params-output-path /app/model-output/params.ini \
--num-epochs 50  --verbosity 1
```

Save state of the container.
```sh
 sudo docker commit DOCKER_CONTAINER_ID image-cnn/trained
```

Run a web-server to perform prediction with a model, which configuration is stored in /app/model-output/params.ini.
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
