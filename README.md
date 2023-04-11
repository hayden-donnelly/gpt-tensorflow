# gpt-tensorflow
Tensorflow implementation of the paper "Improving Language Understanding by Generative Pre-Training," AKA, the original GPT paper.

<img src="./images/gpt-architecture.png" width="270px"></img>

## Tokenization

The ``tokenizer.py`` script offers two different tokenization schemes: spacy tokenization, and character-wise tokenization.
Spacy tokenization is the default, and also the one used in the original paper. However, due to the large vocabulary
size it generates, it may be too memory intensive for some machines. In this case, character-wise tokenization can be used.
To switch to character-wise tokenization, call ``tokenizer.py`` with the argument ``--use_spacy=False``.

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
