# gpt-tensorflow
A Tensorflow implementation of the original GPT.

This code is based on my interpretation of the paper. You can find the authors' original code 
[here](https://github.com/openai/finetune-transformer-lm). Note that I have omitted downstream 
finetuning in this implementation.

<img src="./images/gpt-architecture.png" width="270px"></img>

## Script/Path Assumption
All scripts assume that you will be running them from the root of this repository and will produce errors if
they are not. For example, running ``python scripts/train.py`` from ``gpt-tensorflow`` will not produce errors,
but running ``python train.py`` from ``gpt-tensorflow/scripts`` will. The same goes for any paths you specify
in script arguments. For example, ``data/tiny_shakespeare.txt`` is correct while 
``../data/tiny_shakespeare.txt`` is not.

## Getting Started

1. Build Docker image:
```
bash docker_build.sh
```

2. Start Docker container:
```
bash docker_run.sh
```

3. Tokenize text:
```
python scripts/tokenize.py --no_spacy --text_path data/tiny_shakespeare.txt
```

4. Train a model:
```
python scripts/train.py --num_attention_heads 6 --num_blocks 6 --feed_forward_dim 1024 --batch_size 1
```

5. Sample the trained model (work in progress):
```
python scripts/sample.py --prompt "We know what we are, but know not what we may be."
```

## Tokenization

The ``tokenizer.py`` script offers two different tokenization schemes: spacy tokenization, and character-wise 
tokenization. Spacy tokenization is the default, and also the one used in the original paper. However, due 
to the large vocabulary size it generates, it may be too memory intensive for some machines. In this case, 
character-wise tokenization can be used. To switch to character-wise tokenization, call ``tokenizer.py`` with 
the argument ``--no_spacy``.

Example: 
```
python scripts/tokenizer.py --no_spacy
```

Full list of parameters:
- ``--no_spacy`` : Use character wise tokenizer instead of Spacy tokenizer.
- ``--text_path`` : Path to text file to be tokenized. Default: ``data/tiny_shakespeare.txt``

## Training 

To train the model, call ``scripts/train.py``. All of the model parameters will default to those 
outlined in the original paper, but you can override them by adding ``--<parameter_name> <parameter_value>`` 
arguments when calling ``scripts/train.py``. 

Example: 
```
python scripts/train.py --attention_dim 512
```

Full list of parameters:

- ``--no_spacy`` : Use character wise tokenizer instead of Spacy tokenizer.
- ``--num_blocks`` : Number of transformer blocks. Default: ``12``
- ``--num_attention_heads`` : Number of attention heads in multi-head attention. Default: ``12``
- ``--context_size`` : Number of tokens in each context. Default: ``512``
- ``--attention_dim`` : Dimension of attention layers. Default: ``768``
- ``--feed_forward_dim`` : Dimension of feed forward layers. Default: ``3072``
- ``--activation`` : Activation function. Default: ``gelu``
- ``--token_embed_dim`` : Dimension of token embeddings. Default: ``384``
- ``--dropout`` : Dropout rate. Default: ``0.1``
- ``--epochs`` : Number of epochs. Default: ``100``
- ``--batch_size`` : Batch size. Default: ``64``

For training on a single consumer grade GPU, you'll need to nerf the model a bit. Below are two different configurations that work on an RTX 3070 Ti.

Nerfed batch size (slow training, but model parameters are the same as in the original paper):
```
python scripts/train.py --batch_size 1 --epochs 1
```

Nerfed model (faster training, but model parameters are different from the original paper):
```
python scripts/train.py --num_attention_heads 6 --num_blocks 6 --feed_forward_dim 1024
```

## Docker Container
Building image:
```
bash docker_build.sh
```

Starting container:
```
bash docker_run.sh
```

## Citation

```bibtex
@misc{radford_narasimhan_salimans_sutskever, 
    title={Improving Language Understanding by Generative Pre-Training}, 
    author={Alec Radford and Karthik Narasimhan and Tim Salimans and Ilya Sutskever},
    url={https://openai.com/research/language-unsupervised}
} 
```
