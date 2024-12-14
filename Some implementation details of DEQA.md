| **Experts** | **Pre-trained Models** | **Batch size** | **Epochs** | **Learning rate** | **Twitter2015** | **Twitter2017** |
|:-|:-|:-:|:-:|:-:|:-|:-|
| Text-only, $Q^e$ | *deberta-v3-large* |
| Text and description, $Q^v$ | *deberta-v3-base* | 8 | 10 | $1.0\times10^{-5}$ | 40m 59s | 51m 56s |
| Text and vision, $Q^v$ | *clip-vit-base-patch32* |
| Text-only, $Q^c$ | *deberta-v3-large* | 8 | 10 | $1.0\times10^{-5}$ | 29m 47s | 44m 54s |
| Text and description, $Q^c$ | *deberta-v3-large* | 8 | 10 | $1.0\times10^{-5}$ | 1h 19m 22s | 44m 54s |
| Text and vision, $Q^c$ | *clip-vit-large-patch14-336* | 8 | 16 | $5.0\times10^{-6}$ | 30m 12s | 24m 6s |

Some implementation details of DEQA. We refer to each expert by the modality (and its combinations) used, as well as the type of query employed. Twitter2015 and Twitter2017 refer to the training times on their respective datasets. Since the three experts in the sub-model for MATE are trained jointly, they use the same hyperparameters and do not have separate training times. Therefore, we have annotated the hyperparameters and training time for the sub-model for MATE in the row corresponding to the *Text and description aspect validation expert*.

