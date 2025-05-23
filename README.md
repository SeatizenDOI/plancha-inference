<p align="center">
  <a href="https://doi.org/10.5281/zenodo.15228534"><img src="https://zenodo.org/badge/790058279.svg" alt="DOI"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/graphs/contributors"><img src="https://img.shields.io/github/contributors/SeatizenDOI/plancha-inference" alt="GitHub contributors"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/network/members"><img src="https://img.shields.io/github/forks/SeatizenDOI/plancha-inference" alt="GitHub forks"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/issues"><img src="https://img.shields.io/github/issues/SeatizenDOI/plancha-inference" alt="GitHub issues"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/blob/master/LICENSE"><img src="https://img.shields.io/github/license/SeatizenDOI/plancha-inference" alt="License"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/pulls"><img src="https://img.shields.io/github/issues-pr/SeatizenDOI/plancha-inference" alt="GitHub pull requests"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/stargazers"><img src="https://img.shields.io/github/stars/SeatizenDOI/plancha-inference" alt="GitHub stars"></a>
  <a href="https://github.com/SeatizenDOI/plancha-inference/watchers"><img src="https://img.shields.io/github/watchers/SeatizenDOI/plancha-inference" alt="GitHub watchers"></a>
</p>
<div align="center">
  <a href="https://github.com/SeatizenDOI/plancha-inference">View framework</a>
  ·
  <a href="https://github.com/SeatizenDOI/plancha-inference/issues">Report Bug</a>
  ·
  <a href="https://github.com/SeatizenDOI/plancha-inference/issues">Request Feature</a>
</div>

<div align="center">

# Plancha Inference

</div>

Plancha-inference is used to apply [Jacques](https://github.com/IRDG2OI/jacques) and a multilabel model from hugging-face like [DinoVdeau](https://github.com/SeatizenDOI/DinoVdeau)

This repository works with sessions that contain a metadata.csv file in a METADATA folder. The metadata.csv file needs to have a column name that includes the term "relative_file_path".

At the end of the process, this code will create a PROCESSED_DATA/IA folder that contains prediction and score files in CSV format, as well as raster files of the predictions.

The METADATA folder will also have two additional files, which are the predictions with GPS coordinates and IMU values.

* [Docker](./docker.README.md)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)


## Installation

All the sessions was proceed on this hardware (Dell Precision 7770):

- Intel Core i9-12950HX
- 64.0 Gio RAM
- NVIDIA RTX A3000 12GB Laptop GPU

And the software was :

- Ubuntu 24.04.4 LTS
- Nvidia driver version 570, Cuda 12.8
- nvcc (NVIDIA Cuda compiler driver) : Cuda compilation tools, release 12.8, V12.8.93 Build cuda_12.8.r12.8/compiler.35583870_0
- TensorRT 10.9

To ensure a consistent environment for all users, this project uses a Conda environment defined in a `inference.yml` file. Follow these steps to set up your environment.

I wish you good luck for the installation.


1. **Setup Nvidia Driver:** Please install your [nvidia driver](https://www.nvidia.com/fr-fr/drivers/unix/).

2. **Download and install CudaToolkit:** Check [TensorRT requirements](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to get the good [CudaToolkit](https://developer.nvidia.com/cuda-toolkit).

3. **Install Conda:** If you do not have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

4. **Create the Conda Environment:** Navigate to the root of the project directory and run the following command to create a new environment from the `inference.yml` file:
   ```bash
   conda env create -f inference.yml
   ```

5. **Activate the Environment:** Once the environment is created, activate it using:
   ```bash
   conda activate inference_env
   ```

6. **Install PyTorch:** Install PyTorch. It is recommended to install PyTorch with CUDA support for optimal performance. Follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch with the appropriate options for your system.

Here is an example command to install PyTorch with CUDA support:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

⚠️ Note: At this point, you can already use the repository without TensorRT acceleration.
You do not need to install TensorRT or use -jtrt / -mltrt arguments unless you want to speed up inference.


7. **(Optionnal) Install TensorRT:** Follow this [link](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#).

8. **(Optional) Install additional packages for TensorRT usage:**
```bash
pip install tensorrt cuda-python onnx onnx-graphsurgeon 
```


For understand cuda installation : https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi


## Usage

To run the workflow, navigate to the project root and execute:

```bash
python inference.py [OPTIONS]
```

### Input Parameters

The script allows you to select an input method from several mutually exclusive options:

* `-efol`, `--enable_folder`: Use data from a session folder.
* `-eses`, `--enable_session`: Use data from a single session.
* `-ecsv`, `--enable_csv`: Use data from a CSV file.

### Input Paths

You can specify the paths to the files or folders to be used as input:

* `-pfol`, `--path_folder`: Path to the session folder. Default: /home/bioeos/Documents/Bioeos/plancha-session.
* `-pses`, `--path_session`: Path to a specific session. Default: /media/bioeos/E/202309_plancha_session/20230926_REU-HERMITAGE_ASV-2_01/.
* `-pcsv`, `--path_csv_file`: Path to the CSV file containing the inputs. Default: ./csv_inputs/retry.csv.

### Jacques model parameters

* `-jcku`, `--jacques_checkpoint_url`: default="20240513_v20.0", Specified which checkpoint file to used, if checkpoint file is not found we downloaded it from zenodo. 
* `-jtrt`, `--jacques_trt`: Build an engine from jacques_checkpoint_url, use tensorrt to speedup inference
* `-jcsv`, `--jacques_csv`: Used csv file of jacques predictions
* `-nj`, `--no_jacques`: Didn't used jacques model

### Mulilabel model parameters

* `-mlu`, `--multilabel_url`: Hugging face repository. Default : lombardata/DinoVdeau-large-2024_04_03-with_data_aug_batch-size32_epochs150_freeze
* `-mltrt`, `--multilabel_trt`: Speedup inference with tensorrt
* `-nml`, `--no_multilabel`: Didn't used multilabel model

❗ You can safely omit -jtrt and -mltrt if you do not want to use TensorRT. The inference will still work using regular PyTorch execution.

### Optional Arguments

The script also includes optional arguments to fine-tune its behavior:

* `-np`, `--no-progress`: Hide display progress bar. Default: False
* `-ns`, `--no-save`: Don't save annotations. Default: False
* `-npr`, `--no_prediction_raster`: Don't produce predictions rasters. Default False
* `-c`, `--clean`: Clean pdf preview and predictions files. Default: False
* `-is`, `--index_start`: Choose from which index to start. Default: 0
* `-ip`, `--index_position`: If != -1, take only session at selected index. Default: -1
* `-bs`, `--batch_size`: Numbers of frames processed in one time. Default: 1
* `-minp`, `--min_prediction`: Minimum for keeping predictions after inference. Default: 100

### Example 

An example of command to process. We process a folder of session using tensorrt to speedup inference and we clean old predictions files.
```bash
python inference.py -efol -pfol /path/to/my/folder -mltrt -jtrt -c
```

## Contributing

Contributions are welcome! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes with clear, descriptive messages.
4. Push your branch and submit a pull request.

## License

This framework is distributed under the CC0-1.0 license. See `LICENSE` for more information.

<div align="center">
  <img src="https://github.com/SeatizenDOI/.github/blob/main/images/logo_partenaire.png?raw=True" alt="Partenaire logo" width="700">
</div>
