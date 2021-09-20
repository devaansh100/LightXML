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

## Loss Correction

There seems to be an oversight in the original LightXML paper. It mentions that the loss function would be ```Lg + Ld``` and would be backpropagated from the discriminator. However, since ```Lg``` does not contain any information of the discriminator, it instantly goes to 0 in the first partial derivative. Therefore, all the gradient updates that occur in the generator happen due to the information in ```Ld```. While this is giving good results, it has not reached its full potential, since the generator loss function information is not being used. To fix this, add the generator loss to upstream loss and continue the backpropagation process.