# Blueprint Vectorizer

Official implementation of the paper [Vectorizing Building Blueprints](https://openaccess.thecvf.com/content/ACCV2022/papers/Song_Vectorizing_Building_Blueprints_ACCV_2022_paper.pdf) (**ACCV 2022**)

## Preparation

### Environment

We use conda for Python environment management, which you may run the following code.

```
conda create -n blueprint python=3.7
conda activate blueprint

conda install -y pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
conda install -y matplotlib Pillow scikit-image scikit-learn tensorboard shapely

pip install ruamel.yaml tqdm opencv_python
```

### Data

Unfortunately we are not allowed to share the blueprint images or their labels.
Here we will explain the format and file structure for the input data, so you can prepare yours the same way.

Each floorplan example should have three files:
* Floorplan image in JPEG format
* Pixel-wise ground-truth instance segmentation as one numpy array, where each instance mask receives a unique ID
* Pixel-wise ground-truth semantic segmentation as one numpy array, with the following class mapping

```
class_map = {
  0: 'Background',
  1: 'Outer wall',
  2: 'Inner wall',
  3: 'Window',
  4: 'Door',
  5: 'Open portal',
  6: 'Room',
  7: 'Frame',
}
```

The three files should have the same name aside from the different file extension, and they should be put inside `data/` folder under `fp_img/`, `instance/`, and `semantic/` folders.

We perform 10-fold cross-validation, so you will also need to split your floorplan examples into 10 splits. Each JSON split file should contain an array of just the file names, without the extension.

At a high-level, your folder structure from root should look like this:
```
blueprint-vectorizer
├── 00_preprocess/
├── 01_instance_seg/
├── ...
├── 06_eval/
└── data
    ├── preprocess/
    │   ├── fp_img/
    │   ├── instance/
    │   └── semantic/
    └── splits/
        ├── ids_0.json
        ├── ...
        └── ids_9.json
```

### Pretrained models

Please download them from [here](https://drive.google.com/file/d/1LXHspV6-73tox3_0YluzzTmDg2RQCNCS/view?usp=sharing) and unzip it so `ckpts/` is in the root directory.

## Running the pipeline

Our system consists of a number of pipelines. Each one is trained separately and the intermediate outputs are chained together to obtain the final results.
In each section, we describe how to start the training and obtain the intermediate outputs for the next stage.

Unfortunately even if you just want to run the pretrained models, you still need to prepare the GT segmentation labels. Or you can modify the dataloaders so they do not try to load them.

### Instance segmentation
This stage outputs the instance segmentation of each floorplan.

```
cd 01_instance_seg/
python train.py --test_fold_id 0     # train
python predict_new.py --test_fold 0  # predict
```

### Semantic classification
This stage classifies each predicted instance into one of eight classes.

```
cd 02_type_class/
python data_gen.py                             # data preprocessing
python train.py --config_path example.yaml     # train
python evaluate.py --config_path example.yaml  # predict
```

### Frame detection
This stage learns a classifier that determines the correct topology of the frame symbol around doors and other objects.
Note that frame detection is not evaluated separately; it is to be used for the correction stage.

```
cd 03_frame_detect/
python train_cnn.py --hparam_f example.yaml
```

### Frame correction
This stage trains a Generative Adversarial Network to correct any missing or extra frame symbols.
It also outputs the fixed segmentation masks, using the previously trained topology classifier to detect any incorrect frame topologies.

```
cd 04_frame_correct/
python train_cnn.py --hparam_f example.yaml

GAN_YAML = "../ckpts/04_frame_correct/2021-07-26_gan_xval_00/hparams.yaml"
TOPO_YAML = "../ckpts/03_frame_detect/2021-07-26_topo_net_xval_00/hparams.yaml"
python refinery.py --gan_f $GAN_YAML --topo_f $TOPO_YAML
```

### Heuristic simplification
This stage takes in a segmentation mask and smoothes out the edges so we have a more concise map that we can vectorize.

```
cd 05_simplification/
python actions.py --method rh
```

## Evaluation

Coming soon...

## Contact
Weilian Song, weilians@sfu.ca

## Bibtex
```
@InProceedings{Song_2022_ACCV,
    author    = {Song, Weilian and Abyaneh, Mahsa Maleki and A Shabani, Mohammad Amin and Furukawa, Yasutaka},
    title     = {Vectorizing Building Blueprints},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {1044-1059}
}
```

## Acknowledgment
The research is supported by NSERC Discovery Grants, NSERC Discovery Grants Accelerator Supplements, and DND/NSERC Discovery Grants. We also thank GA Technologies for providing us with the building blue-print images.