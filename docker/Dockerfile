FROM nvcr.io/nvidia/tensorrt:23.12-py3

ARG index_pos
ENV env_index_pos=$index_pos

ARG index_start
ENV env_index_start=$index_start

# Refresh apt.
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean

# Setup python with tensorrt and install all dependencies.
RUN /opt/tensorrt/python/python_setup.sh && \ 
pip install numpy natsort transformers pandas pillow tqdm zenodo_get pypdf reportlab cartopy scipy matplotlib geocube huggingface_hub && \
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy entrypoint script into the container.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Make the entrypoint script executable.
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create new user.
RUN useradd -ms /bin/bash seatizen

# Add local directory.
ADD ../. /home/seatizen/app/

# Setup workdir in directory.
WORKDIR /home/seatizen/app

# Change permission in new directory.
RUN chown -R seatizen:seatizen /home/seatizen

# Change with our user.
USER seatizen

# Define the entrypoint script to be executed.
ENTRYPOINT ["entrypoint.sh"]


