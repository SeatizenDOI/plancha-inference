# Docker cheatsheet

To use this docker image you will need Nvidia driver 550 at least to get cuda 12.4.

You also need [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  

We need torch==2.0 with torchvision==0.17. When selecting torch, you need to match the torch version with your version.

We also need a tensorrt version below 10.

## Build.

`docker build -t plancha-inference:latest .`
`docker tag plancha-inference:latest seatizendoi/plancha-inference:latest`
`docker push seatizendoi/plancha-inference:latest`

```bash
docker build -t plancha-inference:latest . && \
docker tag plancha-inference:latest seatizendoi/plancha-inference:latest && \
docker push seatizendoi/plancha-inference:latest
```

## Run.

`docker run --gpus all --rm -v /media/bioeos/F/202504_plancha_session:/home/seatizen/plancha plancha-inference:latest [OPTIONS OF INFERENCE.PY file]`

```
docker run --gpus all --rm -v /media/bioeos/F/202504_plancha_session:/home/seatizen/plancha plancha-inference:latest -efol -pfol /home/seatizen/plancha -jgpu -mlgpu -c
```

Launch container with :
* `--gpus all`: All gpus found.
* `--rm`: Remove the container after exit.
* `-v /path/to/folder/plancha_sessions:/plancha`: Mount a volume to acces to plancha session.

After the image, you can pass all the arguments of inference.py file. 

# Singularity cheatsheet

Datarmor, the ifremer supercomputer, doesn't handle custom docker image easily. You need to convert your docker image to a singularity container.

## Build container.


singularity build -f inference.sif docker://seatizendoi/plancha-inference:latest

## Launch container.

singularity run --nv --bind /home1/datawork/villien/plancha-session:/home/seatizen/plancha inference.sif