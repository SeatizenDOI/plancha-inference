FROM nvcr.io/nvidia/tensorrt:23.12-py3

ARG index_pos
ENV env_index_pos=$index_pos

ARG index_start
ENV env_index_start=$index_start

# Refresh apt && Setup python with tensorrt and install all dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    /opt/tensorrt/python/python_setup.sh && \ 
    pip install --no-cache-dir numpy natsort transformers pandas pillow tqdm zenodo_get \
    pypdf reportlab cartopy scipy matplotlib geocube huggingface_hub && \
    pip3 install --no-cache-dir torch torchvision \
    torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy entrypoint script into the container.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the entrypoint script executable.
# Create new user.
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    useradd -ms /bin/bash seatizen

# Add local directory.
ADD --chown=seatizen ../. /home/seatizen/app/

# Setup workdir in directory.
WORKDIR /home/seatizen/app

# Change with our user.
USER seatizen

# Define the entrypoint script to be executed.
ENTRYPOINT ["entrypoint.sh"]


