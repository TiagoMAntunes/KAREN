# KAREN: Unifying Hatespeech Detection and Benchmarking

This file includes evaluation results for each models implemented on each of the datasets. When contributing a new model/dataset, please update the results accordingly.

## HateXPlain Results
HateXPlain evaluation. Precision, Recall and F1-score are the results from the hatespeech class.

| Model	| Accuracy	| Precision	| Recall | F1|
| ------| ----------| ----------| -------| --|
| **Bert**| **0.689**| 	**0.777**	| **0.752**| 	**0.764**|
| CNN	| 0.613	| 0.711	| 0.69	| 0.7|
| Softmax Regression	| 0.366	| 0.298	| 0.087	| 0.135|
| RNN	| 0.55	| 0.684| 	0.654| 	0.669|
| BiLSTM| 	0.59| 	0.662| 	0.764| 	0.71|
| NetLSTM	| 0.61| 	0.678| 	0.76| 	0.716|
| GRU	| 0.609	| 0.666	| 0.777	| 0.72|
| Transformer (1 layer)	| 0.486| 	0.495	| 0.6	| 0.543|
| Transformer (2 layers)| 	0.532| 	0.551	| 0.732| 	0.629|
| CharCNN | 0.552 | 0.65 | 0.61 | 0.63 |
| **AngryBERT (primary only)** | **0.649** | **0.736** | **0.695** | **0.764** |
| **DistilBERT** | **0.646** | **0.766** | **0.704** | **0.734** |
| RNN + GloVe	| 0.546| 	0.59	| 0.779	| 0.672 |
| **CNN + GloVe**	| **0.644**	| **0.69** | **0.767**| **0.726**|
| **BiLSTM + GloVe**	| **0.637**| 	**0.677**	| **0.781**| 	**0.73**|
| **GRU + GloVe**| **0.64**| 	**0.699**	| **0.736** | **0.717** |
| NetLSTM + GloVe	| 0.616| 	0.679| 	0.756| 	0.715|
| Transformer (1 layer) + GloVe	| 0.564	| 0.581	| 0.785	| 0.668|
| Transformer (2 layers) + GloVe| 	0.572| 	0.751| 	0.609	| 0.672|
| CharCNN + Glove | 0.573 | 0.631 | 0.753 | 0.686 |
| **AngryBERT + Glove (primary only)** | **0.660** | **0.75** | **0.771** | **0.76** |
| UNet + Glove | 0.602 | 0.714 | 0.670 | 0.691 |
| UNet | 0.548 | 0.646 | 0.670 |  0.657 |
| VDCNN | 0.552 | 0.694 | 0.653 | 0.673 |
| VDCNN + Glove | 0.601 | 0.681 | 0.694 | 0.688 |

## HSAOL Results
HSAOL evaluation. Precision, Recall and F1-score are the results from the hate speech class. Due to the imbalanced nature of this dataset, the results from the hate speech class may be suboptimal.

| Model	| Accuracy	| Precision	| Recall | F1|
| ------| ----------| ----------| -------| --|
| **CNN**	| **0.908** | **0.5**	| **0.218**	| **0.304**	|
| Softmax Regression	| 0.757	| 0	| 0	| 0 |
| RNN	| 0.865	| 0 | 	0 | 	0|
| BiLSTM| 	0.896| 	0 | 	0 | 	0 |
| NetLSTM	| 0.897| 	0.377 | 	0.168| 	0.233|
| GRU	| 0.904	| 0 | 0	| 0 |
| Transformer (1 layer)	| 0.876 | 0.448 | 0.104 | 0.169	|
| Transformer (2 layers)| 0.887 | 0.4	| 0.192 | 0.26 |
| UNet | 0.897 | 0.475 |  0.224  |  0.304 |
| DistilBERT | 0.908 | 0.441  | 0.345| 0.387 |
| AngryBERT (primary only) | 0.908 | 0.3 | 0.024 | 0.044 |
| RNN + GloVe	| 0.898| 	0	| 0 | 0|
| **CNN + GloVe**	| **0.915**	| **0.518** | **0.244** | **0.331** |
| BiLSTM + GloVe	| 0.906 | 	0	| 0 | 	0 |
| GRU + GloVe | 0.909 | 	0	| 0 | 0 |
| NetLSTM + GloVe	| 0.906 | 	0.397 | 	0.193| 	0.260 |
| Transformer (1 layer) + GloVe	| 0.892	| 0.471	| 0.128	| 0.201 |
| Transformer (2 layers) + GloVe| 0.907 | 0.474 | 0.216 | 0.287 |
| UNet + Glove | 0.912 | 0.524  | 0.264 | 0.351 |
| AngryBERT + Glove (primary only) |  0.913 | 0.385 | 0.12 | 0.183 |
| **Bert** | **0.918** | **0.552**	| **0.384** | **0.453** |
| CharCNN |  N/A |  N/A |  N/A |  N/A |
| CharCNN + Glove |  N/A |  N/A |  N/A |  N/A |
| VDCNN + Glove | 0.598 | 0.655 | 0.709 | 0.681 |

## ETHOS Results
ETHOS evaluation. Precision, Recall and F1-score are the results from the hate speech class.

| Model	| Accuracy	| Precision	| Recall | F1|
| ------| ----------| ----------| -------| --|
| CNN	| 0.644 | 0.639	| 0.548	| 0.590	|
| Softmax Regression	| 0.467	| 0.465	| 0.952	| 0.625  |
| RNN	| 0.522	| 0.4 |	0.048 |	0.085|
| BiLSTM| 	0.589| 	0.619 | 	0.310 | 	0.413 |
| NetLSTM	| 0.544| 	0.6 | 	0.071| 	0.128 |
| GRU	| 0.6	| 0.588 | 0.476	| 0.526|
| Transformer (1 layer)	| 0.556 | 0.6 | 0.143 | 0.231	|
| Transformer (2 layers)| 0.478 | 0.468	| 0.881 | 0.612 |
| UNet | 0.478 | 0.471 |0.976  | 0.636 |
| UNet (using linear) | 0.522 |  0.455|0.119  | 0.189 |
| **DistilBERT** | **0.744** |  **0.757** | **0.667** |**0.709** |
| **AngryBERT (primary only)** | **0.711**|**0.674**  |**0.738** | **0.705** |
| RNN + GloVe	| 0.544| 	0.514	| 0.429 | 0.468|
| CNN + GloVe	| 0.644 | 0.639 | 0.548 | 0.590 |
| BiLSTM + GloVe	| 0.656 | 	0.628	| 0.643 | 	0.635 |
| **GRU + GloVe** | **0.711** | 	**0.674**	| **0.738** | **0.705** |
| NetLSTM + GloVe	| 0.644 | 	0.614 | 	0.643| 	0.628 |
| Transformer (1 layer) + GloVe	| 0.611	| 0.569	| 0.690	| 0.624 |
| Transformer (2 layers) + GloVe| 0.489 | 0.463 | 0.595 | 0.521 |
| **UNet + Glove** | **0.722** | **0.707** | **0.690** | **0.699** |
| UNet + Glove (using linear) |0.533 | 0   | 0 |  0|
| **AngryBERT + Glove (primary only)** |  **0.733** | **0.75**  | **0.643** |  **0.692**|
| **Bert** | **0.756** | **0.763**	|  **0.690**| **0.725** |
| CharCNN + GloVe | 0.533 | 	0	| 0 | 0 |
| CharCNN + GloVe	| 0.533 | 	0 | 	0| 	0 |
| VDCNN | 0.5 | 0.459 | 0.405 | 0.430 |
| VDCNN + Glove | 0.755 | 0.708 | 0.810 | 0.756 |
