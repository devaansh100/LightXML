# LightXML

LightXML: Transformer with dynamic negative sampling for High-Performance Extreme Multi-label Text Classiﬁcation
This repository was forked from https://github.com/kongds/LightXML. The main branch has the added functionality of loading a model from checkpoint. The aim of this repository is to implement suggested improvements in the original LightXML model.

## Datasets
LightXML uses the same dataset with AttentionXML and X-Transform.

please download them from the following links.
* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Wiki-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)
   
## Experiments
train and eval
```bash
./run.sh [eurlex4k|wiki31k|amazon13k|amazon670k|wiki500k]
```

## Squeeze and Excitation Block

LightXML concats the last 5 hidden layers of the transformer output to create text embeddings. While the transformer model does have self-attention built in, the squeeze and excitation block adds a second layer of attention between these 5 hidden layers. It allows the text representations to be more expressive since the more relevant layers will be scaled up. Another self-attention layer can be added for each element of the embedding, however, that will increase the model size much more than this block.