## Next Steps 

1. Find out how to add classes to an existing model. (yes, it's called transfer learning)
1.1 See if subclasses are a thing.
2. Read through the transfer learning tutorial.
3. Look for a dataset with the fish class. 
4. Learn how to train a linear classifier. 
A linear classifier is the head (last layer) of the model which is the layer
that outputs the final prediction on what somehting is. For FNF, it makes
sense to train only a linear classifier because our dataset size is small.
If we can find a larger dictionary of fish, it may make sense to fine tune the
DeTR model (modify multiple end layers) beause our dataset is fairly different than 
COCO. Due to our dataset size and difference, it may make sense for us to train
a classifier several layers closer to the start of the network, instead of 
at the last layer. 
5. Use this dataset: https://huggingface.co/datasets/Francesco/fish-market-ggjso