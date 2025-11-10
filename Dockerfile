# Use an official Python runtime as a parent image
FROM python:3.12-slim-bullseye

# Install lib, create user.
RUN apt-get update && apt-get install libexpat1-dev -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir \
    numpy==2.2.4 \
    natsort==8.4.0 \
    transformers==4.51.3 \
    pandas==2.2.3 \
    pillow==11.2.1 \
    tqdm==4.67.1 \
    zenodo_get==1.6.1 \
    pypdf==5.4.0 \
    reportlab==4.3.1 \
    cartopy==0.24.1 \
    scipy==1.15.2 \
    matplotlib==3.10.1 \
    geocube==0.7.1 \
    huggingface_hub==0.30.2 \
    torch==2.6.0 \ 
    torchvision==0.21.0 \
    polars==1.35.1 \
    hf_xet==1.2.0 \
    open-clip-torch==3.2.0 && \
    useradd -ms /bin/bash seatizen

# Add local directory.
ADD --chown=seatizen ../. /home/seatizen/app/

# Setup workdir in directory.
WORKDIR /home/seatizen/app

# Change with our user.
USER seatizen

# Define the entrypoint script to be executed.
ENTRYPOINT ["python", "inference.py"]