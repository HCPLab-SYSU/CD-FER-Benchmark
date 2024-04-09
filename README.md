# Cross Domain Facial Expression Recognition Benchmark

[![1](https://img.shields.io/badge/SOTA-Leaderboard_On_RAF-blue)](https://paperswithcode.com/sota/cross-domain-facial-expression-recognition-on)
[![1](https://img.shields.io/badge/SOTA-Leaderboard_On_AFE-blue)](https://paperswithcode.com/sota/cross-domain-facial-expression-recognition-on-2)

Implementation of papers: 

- [Cross-Domain Facial Expression Recognition: A Unified Evaluation Benchmark and Adversarial Graph Learning](https://ieeexplore.ieee.org/document/9628054)   
  IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI), 2022.   
  Tianshui Chen*, Tao Pu*, Hefeng Wu, Yuan Xie, Lingbo Liu, Liang Lin. (* equally contributed)

- [Adversarial Graph Representation Adaptation for Cross-Domain Facial Expression Recognition](https://dl.acm.org/doi/10.1145/3394171.3413822)   
  ACM International Conference on Multimedia (ACM MM), 2020. (**Oral Presentation**)   
  Yuan Xie, Tianshui Chen, Tao Pu, Hefeng Wu, Liang Lin.
   

![Pipeline](./Images/Pipeline.png)

## Environment
Ubuntu 16.04 LTS, Python 3.5, PyTorch 1.3   

**Note:** We also provide a docker image for this project, [click here](https://hub.docker.com/r/putao3/images/tags). (Tag: py3-pytorch1.3-agra)


## Datasets

To apply for the AFE, please complete the [AFE Database User Agreement](./Agreement.pdf) and submit it to tianshuichen@gmail.com or putao537@gmail.com. 

**Note:** 
1) **The AFE Database Agreement needs to be signed by the faculty member at a university or college and sent it by email.**
2) In order to comply with relevant regulations, you need to apply for the image data of the following data sets by yourself, including [CK+](http://www.jeffcohn.net/wp-content/uploads/2020/10/2020.10.26_CK-AgreementForm.pdf100.pdf.pdf), [JAFFE](https://zenodo.org/record/3451524#.YXdc1hpBw9E), [SFEW 2.0](https://cs.anu.edu.au/few/AFEW.html), [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), [ExpW](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html), [RAF](http://www.whdeng.cn/raf/model1.html). 

## Usage

Before running these script files, you should download datasets and pre-train model, and run **getPreTrainedModel\_ResNet.py (or getPreTrainedModel\_MobileNet.py)**.

### Run ICID
```bash
cd ICID
bash Train.sh
```

### Run DFA
```bash
cd DFA
bash Train.sh
```

### Run LPL
```bash
cd LPL
bash Train.sh
```

### Run DETN 
```bash
cd DETN
bash TrainOnSourceDomain.sh     # Train Model On Source Domain
bash TransferToTargetDomain.sh  # Then, Transfer the Model to the Target Domain
```

### Run FTDNN
```bash
cd FTDNN
bash Train.sh
```

### Run ECAN
```bash
cd ECAN
bash TrainOnSourceDomain.sh     # Train Model On Source Domain
bash TransferToTargetDomain.sh  # Then, Transfer the Model to the Target Domain
```

### Run CADA
```bash
cd CADA
bash TrainOnSourceDomain.sh     # Train Model On Source Domain
bash TransferToTargetDomain.sh  # Then, Transfer the Model to the Target Domain
```

### Run SAFN
```bash
cd SAFN
bash TrainWithSAFN.sh
```

### Run SWD
```bash
cd SWD
bash Train.sh
```

### Run AGRA
```bash
cd AGRA
bash TrainOnSourceDomain.sh     # Train Model On Source Domain
bash TransferToTargetDomain.sh  # Then, Transfer the Model to the Target Domain
```

## Result


### Souce Domain: RAF

| Methods | Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ICID** | ResNet-50 | 74.42 | 50.70 | 48.85 | 53.70 | **69.54** | 59.44 |
| **DFA** | ResNet-50 | 64.26 | 44.44 | 43.07 | 45.79 | 56.86 | 50.88 |
| **LPL** | ResNet-50 | 74.42 | 53.05 | 48.85 | 55.89 | 66.90 | 59.82 |
| **DETN** | ResNet-50 | 78.22 | 55.89 | 49.40 | 52.29 | 47.58 | 56.68 |
| **FTDNN** | ResNet-50 | 79.07 | 52.11 | 47.48 | 55.98 | 67.72 | 60.47 |
| **ECAN** | ResNet-50 | <u>79.77</u> | 57.28 | 52.29 | 56.46 | 47.37 | 58.63 |
| **CADA** | ResNet-50 | 72.09 | 52.11 | <u>53.44</u> | <u>57.61</u> | 63.15 | 59.68 |
| **SAFN** | ResNet-50 | 75.97 | <u>61.03</u> | 52.98 | 55.64 | 64.91 | <u>62.11</u> |
| **SWD** | ResNet-50 | 75.19 | 54.93 | 52.06 | 55.84 | 68.35 | 61.27 |
| **Ours** | ResNet-50 | **85.27** | **61.50** | **56.43** | **58.95** | <u>68.50</u> | **66.13** |

---

| Methods | Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ICID** | ResNet-18 | 67.44 | 48.83 | 47.02 | 53.00 | <u>68.52</u> | 56.96 |
| **DFA** | ResNet-18 | 54.26 | 42.25 | 38.30 | 47.88 | 47.42 | 46.02 |
| **LPL** | ResNet-18 | 72.87 | 53.99 | 49.31 | 53.61 | 68.35 | 59.63 |  
| **DETN** | ResNet-18 | 64.19 | 52.11 | 42.25 | 42.01 | 43.92 | 48.90 |
| **FTDNN** | ResNet-18 | <u>76.74</u> | 50.23 | 49.54 | 53.28 | 68.08 | 59.57 |
| **ECAN** | ResNet-18 | 66.51 | 52.11 | 48.21 | 50.76 | 48.73 | 53.26 |
| **CADA** | ResNet-18 | 73.64 | <u>55.40</u> | <u>52.29</u> | <u>54.71</u> | 63.74 | <u>59.96</u> |
| **SAFN** | ResNet-18 | 68.99 | 49.30 | 50.46 | 53.31 | 68.32 | 58.08 |
| **SWD** | ResNet-18 | 72.09 | 53.52 | 49.31 | 53.70 | 65.85 | 58.89 |
| **Ours** | ResNet-18 | **77.52** | **61.03** | **52.75** | **54.94** | **69.70** | **63.19** |

---

| Methods | Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ICID** | MobileNet V2 | 57.36 | 37.56 | 38.30 | 44.47 | 60.64 | 47.67 |
| **DFA** | MobileNet V2 | 41.86 | 35.21 | 29.36 | 42.36 | 43.66 | 38.49 |
| **LPL** | MobileNet V2 | 59.69 | 40.38 | 40.14 | 50.13 | 62.26 | 50.52 |
| **DETN** | MobileNet V2 | 53.49 | 40.38 | 35.09 | 45.88 | 45.26 | 44.02 |
| **FTDNN** | MobileNet V2 | <u>71.32</u> | 46.01 | <u>45.41</u> | 49.96 | <u>62.87</u> | 55.11 |
| **ECAN** | MobileNet V2 | 53.49 | 43.08 | 35.09 | 45.77 | 45.09 | 44.50 |
| **CADA** | MobileNet V2 | 62.79 | 53.05 | 43.12 | 49.34 | 59.40 | 53.54 |
| **SAFN** | MobileNet V2 | 66.67 | 45.07 | 40.14 | 49.90 | 61.40 | 52.64 |
| **SWD** | MobileNet V2 | 68.22 | <u>55.40</u> | 43.58 | <u>50.30</u> | 60.04 | <u>55.51</u> |
| **Ours** | MobileNet V2 | **72.87** | **55.40** | **45.64** | **51.05** | **63.94** | **57.78** |

---

### Souce Domain: AFE

| Methods | Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ICID** | ResNet-50 | 56.59 | 57.28 | 44.27 | 46.92 | 52.91 | 51.59 |
| **DFA** | ResNet-50 | 51.86 | 52.70 | 38.03 | 41.93 | 60.12 | 48.93 |
| **LPL** | ResNet-50 | 73.64 | 61.03 | 49.77 | 49.54 | 55.26 | 57.85 |
| **DETN** | ResNet-50 | 56.27 | 52.11 | 44.72 | 42.17 | 59.80 | 51.01 |
| **FTDNN** | ResNet-50 | 61.24 | 57.75 | 47.25 | 46.36 | 52.89 | 53.10 |
| **ECAN** | ResNet-50 | 58.14 | 56.91 | 46.33 | 46.30 | 61.44 | 53.82 |
| **CADA** | ResNet-50 | 72.09 | 49.77 | <u>50.92</u> | <u>50.32</u> | <u>61.70</u> | 56.96 |
| **SAFN** | ResNet-50 | <u>73.64</u> | <u>64.79</u> | 49.08 | 48.89 | 55.69 | <u>58.42</u> |
| **SWD** | ResNet-50 | 72.09 | 61.50 | 48.85 | 48.83 | 56.22 | 57.50 |
| **Ours** | ResNet-50 | **78.57** | **65.43** | **51.18** | **51.31** | **62.71** | **61.84** |

---

| Methods | Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ICID** | ResNet-18 | 54.26 | 51.17 | 47.48 | 46.44 | 54.85 | 50.84 |
| **DFA** | ResNet-18 | 35.66 | 45.82 | 34.63 | 36.88 | <u>62.53</u> | 43.10 |
| **LPL** | ResNet-18 | 67.44 | **62.91** | 48.39 | 49.82 | 54.51 | 56.61 |   
| **DETN** | ResNet-18 | 44.19 | 47.23 | 45.46 | 45.39 | 58.41 | 48.14 |
| **FTDNN** | ResNet-18 | 58.91 | 59.15 | 47.02 | 48.58 | 55.29 | 53.79 |
| **ECAN** | ResNet-18 | 44.19 | 60.56 | 43.26 | 46.15 | 62.52 | 51.34 |
| **CADA** | ResNet-18 | 72.09 | 53.99 | 48.39 | 48.61 | 58.50 | 56.32 |
| **SAFN** | ResNet-18 | 68.22 | <u>61.50</u> | 50.46 | 50.07 | 55.17 | 57.08 |
| **SWD** | ResNet-18 | <u>77.52</u> | 59.15 | <u>50.69</u> | <u>51.84</u> | 56.56 | <u>59.15</u> |
| **Ours** | ResNet-18 | **79.84** | 61.03 | **51.15** | **51.95** | **65.03** | **61.80** |

---

| Methods | Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ICID** | MobileNet V2 | 55.04 | 42.72 | 34.86 | 39.94 | 44.34 | 43.38 |
| **DFA** | MobileNet V2 | 44.19 | 27.70 | 31.88 | 35.95 | 61.55 | 40.25 |  
| **LPL** | MobileNet V2 | 69.77 | 50.23 | 43.35 | 45.57 | 51.63 | 52.11 |
| **DETN** | MobileNet V2 | 57.36 | 54.46 | 32.80 | 44.11 | **64.36** | 50.62 |
| **FTDNN** | MobileNet V2 | 65.12 | 46.01 | <u>46.10</u> | 46.69 | 53.02 | 51.39 |
| **ECAN** | MobileNet V2 | <u>71.32</u> | **56.40** | 37.61 | 45.34 | <u>64.00</u> | <u>54.93</u> |
| **CADA** | MobileNet V2 | 70.54 | 45.07 | 40.14 | 46.72 | 54.93 | 51.48 |
| **SAFN** | MobileNet V2 | 62.79 | 53.99 | 42.66 | 46.61 | 52.65 | 51.74 |
| **SWD** | MobileNet V2 | 64.34 | 53.52 | 44.72 | **50.24** | 55.85 | 53.73 |
| **Ours** | MobileNet V2 | **75.19** | <u>54.46</u> | **47.25** | <u>47.88</u> | 61.10 | **57.18** |


### Mean of All Methods

#### Souce Domain: RAF

| Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ResNet-50** | 75.87 | 54.30 | 54.49 | 54.82 | 62.09 | 59.51 |
| **ResNet-18** | 69.43 | 51.88 | 47.94 | 51.72 | 61.26 | 56.45 |  
| **MobileNet V2** | 60.78 | 45.15 | 39.59 | 47.92 | 56.46 | 49.98 |

#### Souce Domain: AFE

| Backbone | CK+ | JAFFE | SFEW2.0 | FER2013 | ExpW | Mean |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **ResNet-50** | 65.41 | 57.93 | 47.04 | 47.26 | 57.87 | 55.10 |
| **ResNet-18** | 60.23 | 56.25 | 46.95 | 47.57 | 58.34 | 53.87 |
| **MobileNet V2** | 63.57 | 48.46 | 40.14 | 44.91 | 56.34 | 50.68 |

## Citation

```
@article{Chen2022CD-FER,
  author={Chen, Tianshui and Pu, Tao and Wu, Hefeng and Xie, Yuan and Liu, Lingbo and Lin, Liang},
  title={Cross-Domain Facial Expression Recognition: A Unified Evaluation Benchmark and Adversarial Graph Learning}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  volume={44},
  number={12},
  pages={9887-9903},
  year={2022},
  publisher={IEEE},
  doi={10.1109/TPAMI.2021.3131222}
}

@inproceedings{Xie2020AGRA,
  author={Xie, Yuan and Chen, Tianshui and Pu, Tao and Wu, Hefeng and Lin, Liang},
  title={Adversarial graph representation adaptation for cross-domain facial expression recognition},
  booktitle={Proceedings of the 28th ACM international conference on Multimedia},
  year={2020},
  pages={1255--1264},
  publisher={Association for Computing Machinery},
  doi={10.1145/3394171.3413822}
}
```

## Contributors
For any questions, feel free to open an issue or contact us:    

* tianshuichen@gmail.com
* putao537@gmail.com
* phoenixsysu@gmail.com
