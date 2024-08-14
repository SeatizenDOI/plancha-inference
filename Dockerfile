FROM nvcr.io/nvidia/tensorrt:23.12-py3

ARG index_pos
ENV env_index_pos $index_pos

ARG index_start
ENV env_index_start $index_start

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean

RUN /opt/tensorrt/python/python_setup.sh && \ 
pip install numpy natsort transformers pandas pillow tqdm zenodo_get pypdf reportlab cartopy scipy matplotlib geocube huggingface_hub && \
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124


ADD . /app/

WORKDIR /app

# Copy entrypoint script into the container
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/entrypoint.sh

# Define the entrypoint script to be executed
ENTRYPOINT ["entrypoint.sh"]
# ENTRYPOINT python inference.py -efol -pfol /plancha -jgpu -mlgpu -c

## Build
# docker build -t plancha-inference-image .

## Run 
#docker run --gpus all -it --rm -v /home/gouderg/Documents/Ifremer/plancha:/plancha IMAGE_ID
#docker run --gpus all -it --rm --env index_pos=1 -v /home/gouderg/Documents/Ifremer/plancha:/plancha IMAGE_ID
#docker run --gpus all -it --rm --env index_start=0 -v  /home/gouderg/Documents/Ifremer/plancha:/plancha IMAGE_ID

