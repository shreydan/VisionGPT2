# VisionGPT-2 : Image Captioning Model

![](https://i.imgur.com/dt66Ixl.png)
*that's my dog Sumo, the model has never seen him before :)*

I wanted to build multimodal models for a while now and what better way that to start with Image Captioning, which is kinda like the hello world of multimodal.

### Notebook:

- Kaggle: [📸️✏️ VisionGPT2 Image Captioning | PyTorch 🔥️](https://www.kaggle.com/code/shreydan/visiongpt2-image-captioning-pytorch)
- [Notebook](./visiongpt2-image-captioning-pytorch.ipynb)

### Model

I used the following 2 models:
  - ViT Base, patch size = 16, image size = 224
  - GPT2 small

- I prepared the architecture *almost* from scratch 
- I extracted the useful ViT layers from the `timm` package and used it as the encoder with the pretrained weights.
- As for GPT2, I coded the entirety from scratch, added a new `Cross Attention` layer in the decoder block to get a standard `encoder-decoder` transformer. 
- GPT2 weights were loaded via HuggingFace. Refer to [NanoGPT](https://github.com/karpathy/nanoGPT/).

![](https://i.imgur.com/fk68DMo.jpeg)

### Dataset

The dataset I used was `COCO 2017` with options for `Flickr30k` and `Flickr8k`.

- The dataset preparation was also done from scratch
- my code goes in detail about how to prepare the labels for causal language modeling, calculating the loss while ignoring special tokens, etc.
- Dynamic padding with custom collate function to pad sequences based on the batch and not the max length of the model.


### Training

- The training loop was written from scratch, the metric I used was `perplexity = e^loss`
- I trained it with mixed-precision fp16 using `torch.amp`.
- I initially trained the randomly initialized cross-attention layers, then in further  epochs, I finetuned the entire GPT2 and in further epochs I finetuned the entire ViT-GPT2 model.

### Generation

- Standard `torch.multinomial` sampling based generation with temperature control.
- Support for deterministic generation with `torch.argmax`
- The results are good not great, I only trained on about 30% of the training samples in COCO.

### Results

| Epoch | Train Loss | Train Perplexity | Val Loss | Val Perplexity |
|---|---|---|---|---|
| 0 | 5.164732 | 174.990611 | 3.288565 | 26.804375 |
| 1 | 2.668888 | 14.423919 | 2.341017 | 10.391795 |
| 2 | 2.30841 | 10.058415 | 2.201064 | 9.034617 |
| 3 | 2.033982 | 7.64447 | 2.099659 | 8.163385 |
| 4 | 1.855595 | 6.395501 | 2.08667 | 8.058035 |

### Predictions

![](https://i.imgur.com/YLUoOfS.png)
![](https://i.imgur.com/K1L3OpS.png)
![](https://i.imgur.com/oYm0Hkd.png)
![](https://i.imgur.com/Qr6ua1Z.png)
![](https://i.imgur.com/FErAT56.png)
![](https://i.imgur.com/sVwUgqC.png)
![](https://i.imgur.com/spO5vyl.png)

#### See more [here](https://www.kaggle.com/code/shreydan/visiongpt2-image-captioning-pytorch#Predictions)

---

```Psalm 32:8```