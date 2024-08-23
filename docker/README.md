# Docker cheatsheet

To use this docker image you will need Nvidia driver 550 at least to get cuda 12.4.

You also need [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  

We need torch==2.0 with torchvision==0.17. When selecting torch, you need to match the torch version with your version.

We also need a tensorrt version below 10.

## Command.

The command execute in the entrypoint is :

`python inference.py -efol -pfol /home/seatizen/plancha -jgpu -mlgpu -c`


## Build.

`docker build -t plancha-inference-image:latest -f ./docker/Dockerfile .`
`docker tag plancha-inference-image:latest groderg/plancha-inference-image:latest`
`docker push groderg/plancha-inference-image:latest`

```bash
docker build -t plancha-inference-image:latest -f ./docker/Dockerfile . && \
docker tag plancha-inference-image:latest groderg/plancha-inference-image:latest && \
docker push groderg/plancha-inference-image:latest
```


## Run.

`docker run --gpus all -it --rm -v /home/gouderg/Documents/Ifremer/plancha:/home/seatizen/plancha plancha-inference-image:latest`

`docker run --gpus all -it --rm --env index_pos=3 -v /home/bioeos/Documents/Bioeos/plancha-session:/home/seatizen/plancha plancha-inference-image:latest`

`docker run --gpus all -it --rm --env index_start=0 -v  /home/gouderg/Documents/Ifremer/plancha:/home/seatizen/plancha plancha-inference-image:latest`

Launch container with :
* `--gpus all`: All gpus found.
* `-it`: In interaction mode. You can write command in the docker after the script end.
* `--rm`: Remove the container after exit.
* `-v /path/to/folder/plancha_sessions:/plancha`: Mount a volume to acces to plancha session.

You can add two arguments :
* `--env index_pos=1`: Execute only one session in the folder at index_position. The first is one not zero.
* `--env index_start=0`: Execute all session after index_start. Ex: if index_start = 2, folders_of_sessions[2:]

If arguments is invalid, execute scripts on all sessions.

# Singularity cheatsheet

Datarmor, the ifremer supercomputer, doesn't handle custom docker image easily. You need to convert your docker image to a singularity container.

## Build container.


singularity build -f inference.sif docker://groderg/plancha-inference-image:latest

## Launch container.

singularity run --nv --env index_pos=2 --bind /home1/datawork/villien/plancha-session:/home/seatizen/plancha inference.sif