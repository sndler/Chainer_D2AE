# Chainer-D2AE

This is a reimplementation of [D2AE](https://arxiv.org/abs/1804.03487) in Chainer.
The paper title is Exploring Disentangled Feature Representation Beyond Face Identification, which is presented at CVPR 2018 by Liu et al.

## Training
The step of training D2AE model is separated into two steps.
#### Pretrain for classification of human identity.
#### Train for generation of disentangled features.
We provide a pretrained model on this link and you can skip the pretrain step using it.
A main code for training of the second step is `main.py` script.
You can train it by:
```bash
python main.py rgb 
```

To train a new model, use the `main.py` script.

The command to reproduce the original TSN experiments of RGB modality can be 

```bash
python main.py rgb 
```

For flow models:

```bash
python main.py flow 
```


## Testing

No test code because epick kitchen dataset doesn't have true labels for test data.
