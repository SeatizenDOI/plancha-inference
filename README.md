# Hugging sunnith

Inference to apply jacques and hugging face model

## Jacques model

Jacques model is adapted from [Jacques](https://github.com/IRDG2OI/jacques)

## Logique de ce repository

On choisit la méthode d'import des images
- Importer toutes les sessions depuis un fichier csv ayant la nomenclature suivante

|root_folder|session_name|
|:---|:----|
|/media/bioeos/F/202210_plancha_session | 20221019_SYC-aldabraARM01_ASV-01_00 |
|/media/bioeos/F/202210_plancha_session | 20221019_SYC-aldabraARM01_ASV-01_01 |

- Importer des images depuis un fichier csv ayant la nomenclature suivante <!-- WIP -->

- Importer des sessions depuis un dossier de sessions

- Importer une seule session

Puis on choisit de quelle façon on veut utiliser jacques. Par défaut, on utilise le modèle directe. Sinon, on peut charger un fichier csv contenant le nom des images et si oui ou non elle on la classe 'Useless'

On peut choisir aussi de ne pas utiliser jacques.

Lorsqu'on lance l'inférence avec le modèle jacques, un fichier csv sera crée en sortie.

# Cheatsheet

```bash
    python -m cProfile -s "time" inference.py -eses > profiling/out.test
    python inference.py -eses
```


# Inférence stat 04/02/2024

Malgré les move to cuda, le modèle ne semble pas utilisé la carte graphique
Only with capture image, without converting to rgb, we go at 246 it/s
Only with capture image, converting to rgb, we go at 45 it/s

With jacques_models, without rgb, without predictions, we go at 20 it/s
With jacques_models, without rgb, we go at 13 it/s

With jacques_models, with rgb, we go at 11.5 it/s

Les résultats pour jacques sont exactement les mêmes avec ou sans convert("RGB")

With jacques_csv, without rgb, we go at 220 it/s (not enough images to)
With jacques_csv, with rgb, we go at 37 it/s


# Installation guide pour tensorrt 

Actuellement le code tourne avec 
- tensorrt 8.6.1
- cuda 12.1
- cudnn 8.9.7

Tout d'abord, il faut installer le driver nvidia pour sa machine : https://www.nvidia.com/fr-fr/drivers/unix/

Ensuite il faut installer une version de cuda qui est inférieur ou égale à la version indiqué dans la commande nvidia-smi

Puis on installe cudnn

puis on installe tensorrt

Pour utiliser optimum il faut installer onnxruntime_gpu mais il a une version spéciale si > à 12.1 https://onnxruntime.ai/docs/install/

```bash
   pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ 
```


# CMD to try benchmark

Réaliser avec 60 images ~1.2Mo 3840x2160 pixels

```bash
python inference.py -eses -bs 4 -nj -nml            => 0.0s
python inference.py -eses -bs 4 -nj                 => 21.0s
python inference.py -eses -bs 4 -nj -mlgpu          => 12.0s

python inference.py -eses -bs 4 -nml                => 6.0s
python inference.py -eses -bs 4                     => 22.0s
python inference.py -eses -bs 4 -mlgpu              => 18.0s

python inference.py -eses -bs 4 -jcsv -nml          => 0.0s
python inference.py -eses -bs 4 -jcsv               => 22.0s
python inference.py -eses -bs 4 -jcsv -mlgpu        => 12.0s

python inference.py -eses -bs 4 -jgpu -nml          => 4.0s
python inference.py -eses -bs 4 -jgpu               => 21.0s 
python inference.py -eses -bs 4 -jgpu -mlgpu        => 14.0s
```