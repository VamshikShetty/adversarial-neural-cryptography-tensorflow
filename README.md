# Adversarial Neural Cryptography

![alt text](https://github.com/VamshikShetty/adversarial-neural-cryptography-tensorflow/blob/master/images/medium-symm-crypto.jpg)

orginal paper : ![Learning to protect communications with adversarial neural cryptography](https://arxiv.org/abs/1610.06918)

medium article ![link](https://medium.com/@vamshikdshetty) 

### Training parameters:
```python

learning_rate   = 0.0008
batch_size      = 4096
sample_size     = 4096*5 # 4096 according to the paper
epochs          = 10000  # 850000 according to the paper
steps_per_epoch = int(sample_sizebatch_size)


# Input and output configuration.
TEXT_SIZE = 16
KEY_SIZE  = 16

# training iterations per actors.
ITERS_PER_ACTOR = 1
EVE_MULTIPLIER = 2  # Train Eve 2x for every step of Alice/Bob

```


### Training Loss:
First 300 epochs

![alt text](https://github.com/VamshikShetty/adversarial-neural-cryptography-tensorflow/blob/master/images/0-300.PNG)

Last 300 epochs

![alt text](https://github.com/VamshikShetty/adversarial-neural-cryptography-tensorflow/blob/master/images/2700-3000.PNG)
