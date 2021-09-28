# LightXML

LightXML: Transformer with dynamic negative sampling for High-Performance Extreme Multi-label Text Classiﬁcation. This repository was forked from https://github.com/kongds/LightXML. The main branch has the added functionality of loading a model from checkpoint. The aim of this repository is to implement suggested improvements in the original LightXML model.

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
## Suggested Improvements

### Squeeze and Excitation Block

LightXML concats the last 5 hidden layers of the transformer output to create text embeddings. While the transformer model does have self-attention built in, the squeeze and excitation block adds a second layer of attention between these 5 hidden layers. It allows the text representations to be more expressive since the more relevant layers will be scaled up. Another self-attention layer can be added for each element of the embedding, however, that will increase the model size much more than this block.

Implementation of the same can be viewed in the ```squeeze_and_excitation``` branch.

Since the number of channels is less, reducing the dimenions to 2 will not be very expressive. For that, we can perform some convolutions to increase the number of channels to accentuate the channel attention.

### Loss Correction

There seems to be an oversight in the original LightXML paper. It mentions that the loss function would be ```Lg + Ld``` and would be backpropagated from the discriminator. However, since ```Lg``` does not contain any information of the discriminator, it instantly goes to 0 in the first partial derivative. Therefore, all the gradient updates that occur in the generator happen due to the information in ```Ld```. While this is giving good results, it has not reached its full potential, since the generator loss function information is not being used. To fix this, add the generator loss to upstream loss and continue the backpropagation process.

Implementation of the same can be viewed in the ```loss_correction``` branch.

### Negative Sampling

Eventually, the top b clusters for any instance would not change very much due to the stochastic weight averaging reducing the gradient magnitudes and stabilisation of the model. To keep the proces of dynamic negative sampling vcontinuing, we can randomly choose and add a fixed number of negative labels to each instance. The same was implemented in the Astec algorithm.

### Document Representation

Currently, the document representation is done solely through the [CLS] token. This does not utilise the full capacity of BERT, ie, the token representations are being wasted.. We can use the combination block as suggested in DECAF for the discriminator. The token representations can be added and finally combined with the sentence embedding. If the tokens increase the loss of the model, then during training, their weight vector will naturally be pushed to towards zero.

Taking inspiration from Bonsai, the combination block can be extended to also include representation of co-occuring labels. As before, if this increases the loss, it's weight vector will be pushed towards zero.

This block can either be added only in the classifier, or it can be used to create the Document representation itself.