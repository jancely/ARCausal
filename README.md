# ARCausal

The repo is the official implementation for the paper: [ARCausal:  Causal Mask in Transformer via Vector Autoregressive Learning for Multivariate Time Series Forecasting].


## Installation 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

## Training Example
Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with ARCausal
bash ./scripts/ARCausal_ETTh1.sh
bash ./scripts/ARCausal_ETTh2.sh
bash ./scripts/ARCausal_ETTm1.sh
bash ./scripts/ARCausal_ETTm2.sh
bash ./scripts/ARCausal_ECL.sh
bash ./scripts/ARCausal_Traffic.sh
bash ./scripts/ARCausal_WTH.sh
bash ./scripts/ARCausal_Flight.sh
bash ./scripts/ARCausal_Exchange.sh
```

## Main Result of Multivariate Forecasting

We evaluate the ARCausal on challenging multivariate forecasting benchmarks (**generally hundreds of variates**). **Comprehensive good performance** (MSE/MAE) is achieved.


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- iTransformer (https://github.com/thuml/iTransformer)
- Causal-TSF (https://github.com/AKAGB/Causal-TSF)
- Mamba (https://github.com/state-spaces/mamba)
- Crossformer (https://github.com/Thinklab-SJTU/Crossformer)
- PatchTST (https://github.com/yuqinie98/PatchTST)
- MSGNet (https://github.com/YoZhibo/MSGNet)
- DLinear (https://github.com/ioannislivieris/DLinear)
- TimeMixer (https://github.com/kwuking/TimeMixer)
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)

This work was supported in part by  the Science and Technology Talents and Platform Plan: Technology Innovation Center of Yunnan Province for Digital Water Engineering under Grant 202305AK34003 and in part by the Yunnan Zhang Zongliang Scientist Studio under Grant 2021KXJGZS01.

## Contact

If you have any questions or want to use the code, feel free to contact:
* Chengli Zhou (chenglizhou@mail.ynu.edu.cn)
* Chunna Zhao (zhaochunna@ynu.edu.cn)
