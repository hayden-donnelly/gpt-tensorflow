# gpt-tensorflow
Tensorflow implementation of the paper "Improving Language Understanding by Generative Pre-Training," AKA, the original GPT paper.

This code is based on my interpretation of the paper. You can find the authors' original code [here](https://github.com/openai/finetune-transformer-lm).

<img src="./images/gpt-architecture.png" width="270px"></img>

## Tokenization

The ``tokenizer.py`` script offers two different tokenization schemes: spacy tokenization, and character-wise tokenization.
Spacy tokenization is the default, and also the one used in the original paper. However, due to the large vocabulary
size it generates, it may be too memory intensive for some machines. In this case, character-wise tokenization can be used.
To switch to character-wise tokenization, call ``tokenizer.py`` with the argument ``--use_spacy False``.

Example: 
```
python tokenizer.py --use_spacy False
```

Full list of parameters:
- ``--use_spacy`` : If true, use Spacy tokenizer instead of character wise tokenization. Default: ``True``
- ``--text_path`` : Path to text file to be tokenized. Default: ``../data/tiny_shakespeare.txt``

## Training 

To train the model, call ``train.py``. All of the model parameters will default to those outlined in the original paper, but you can override them by adding ``--<parameter_name> <parameter_value>`` arguments when calling ``train.py``. 

Example: 
```
python train.py --attention_dim 512
```

Full list of parameters:

- ``--use_spacy`` : If true, use Spacy tokenizer instead of character wise tokenization. Default: ``True``
- ``--num_blocks`` : Number of transformer blocks. Default: ``12``
- ``--num_attention_heads`` : Number of attention heads in multi-head attention. Default: ``12``
- ``--context_size`` : Number of tokens in each context. Default: ``512``
- ``--attention_dim`` : Dimension of attention layers. Default: ``768``
- ``--feed_forward_dim`` : Dimension of feed forward layers. Default: ``3072``
- ``--activation`` : Activation function. Default: ``gelu`
- ``--token_embed_dim`` : Dimension of token embeddings. Default: ``384``
- ``--dropout`` : Dropout rate. Default: ``0.1``
- ``--epochs`` : Number of epochs. Default: ``100``
- ``--batch_size`` : Batch size. Default: ``64``

For training on a single consumer grade GPU, you'll need to nerf the model a bit. Below are two different configurations that work on an RTX 3070 Ti.

Nerfed batch size (slow training, but model parameters are the same as in the original paper):
```
python train.py --batch_size 1 --epochs 1
```

Nerfed model (faster training, but model parameters are different from the original paper):
```
python train.py --use_spacy False --num_attention_heads 6 --num_blocks 6 --feed_forward_dim 1024
```

## Docker Environment
Building image:
```
docker-compose build
```

Starting container/environment:
```
docker-compose up -d
```

Opening a shell in container:
```
docker-compose exec gpt-tensorflow bash
```

Instead of opening a shell, you can also go to http://localhost:8888/ to access a Jupyter Lab instance running inside the container.

Stopping container/environment:
```
docker-compose down
```

## Citations

```bibtex
@misc{radford_narasimhan_salimans_sutskever, 
    title={Improving Language Understanding by Generative Pre-Training}, 
    author={Alec Radford and Karthik Narasimhan and Tim Salimans and Ilya Sutskever},
    url={https://openai.com/research/language-unsupervised}
} 
```
