# Semantic-Question-Similarity

The official implementation of our paper: [Tha3aroon at NSURL-2019 Task 8: Semantic Question Similarity in Arabic](paper-url), which was part of [NSURL-2019](http://nsurl.org/tasks/task8-semantic-question-similarity-in-arabic/) workshop on [Task 8](https://www.kaggle.com/c/nsurl-2019-task8) for Arabic Semantic Question Similarity.


## 0. Prerequisites
- Python >= 3.6
- Install required packages listed in `requirements.txt` file
    - `pip install -r requirements.txt`
- To use ELMo embeddings:
  - Clone ELMoForManyLangs repository
    - `git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git`
  - Install the package:
    - `cd ELMoForManyLangs`
    - `python setup.py install`
    - `cd ..`
  - Download and unzip Arabic pre-trainled ELMo model
    - `wget http://vectors.nlpl.eu/repository/11/136.zip -O elmo_dir/136.zip`
    - `unzip elmo_dir/136.zip -d elmo_dir`
    - `cp ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json elmo_dir/cnn_50_100_512_4096_sample.json`

## 1. Data Preprocessing
Data preprocessing step to separate punctuations from words
```
python 1_preprocess.py --dataset-split train
python 1_preprocess.py --dataset-split test
```

## 2. Data Enlarging
Enlarging the data using both Positive and Negative Transitive properties (descriped in the [paper](paper-url))
```
python 2_enlarge.py
```

## 3. Generating Words Embeddings
To make the training step faster, we pre-generate words embeddings from either *ELMo* or *BERT* models and store them in a pickle file
```
python 3_build_embeddings_dict.py --embeddings-type elmo # For ELMo
python 3_build_embeddings_dict.py --embeddings-type bert # For BERT
```
**We adopted using [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs) over [bert-embedding](https://github.com/imgarylai/bert-embedding) because it yields better results.**

## 4. Model Training
Training the model using ELMo with 0.2 dropout, 256 batch size, 100 epochs and 2000 dev set size
```
python 4_train.py --embeddings-type elmo --dropout-rate 0.2 --batch-size 256 --epochs 100 --dev-split 2000
```
**This hyperparameters setup gives the best results according to our experiments, change the values in order to experiment more..**

## 5. Model Inferencing
Inferencing predictions for the test set is done given the path to a certain checkpoint, the default threshold is `0.5` which can be changed using the optional argument `--threshold`
```
python 5_infer.py --model-path checkpoints/epoch100.h5
```

## Model Structure

The following figure illustrates our best model structure.
<p align="center">
  <img src="plots/model_representation.png">
</p>

#### Note: All codes in this repository are tested on [Ubuntu 18.04](http://releases.ubuntu.com/18.04)

## Contributors
1. [Ali Hamdi Ali Fadel](https://github.com/AliOsm).<br/>
2. [Ibraheem Tuffaha](https://github.com/IbraheemTuffaha).<br/>
3. [Mahmoud Al-Ayyoub](https://github.com/malayyoub).<br/>

## License
The project is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
