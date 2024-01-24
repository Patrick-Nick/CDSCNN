# CDSCNN

# Complex-Valued Depthwise Separable Convolutional Neural Network for Automatic Modulation Classification

Code for "Complex-Valued Depthwise Separable Convolutional Neural Network for Automatic Modulation Classification". [[Paper]](https://ieeexplore.ieee.org/abstract/document/10198896)

[Chenghong Xiao](https://orcid.org/0009-0002-5841-7039), [Shuyuan Yang](https://web.xidian.edu.cn/syyang/), *Senior Member, IEEE* and [ZhixiFeng](https://faculty.xidian.edu.cn/FZX/zh_CN/index.htm), *Member, IEEE*

***Abstract***—Automatic modulation classification (AMC) is a critical task in industrial cognitive communication systems. Existing state-of-the-art methods, typified by real-valued convolutional neural networks, have introduced innovative solutions for AMC. However, such models viewed the two constituent components of complex-valued modulated signals as discrete real-valued inputs, causing structural phase damage to original signals and reduced interpretability of the model. In this article, a novel end-to-end AMC model called a complex-valued depthwise separable convolutional neural network (CDSCNN) is proposed, which adopts complex-valued operation units to enable automatic complex-valued feature learning specifically tailored for AMC. Considering the limited hardware resources available in industrial scenarios, complex-valued depthwise separable convolution (CDSC) is designed to strike a balance between classification accuracy and model complexity. With an overall accuracy (OA) of 62.63% on the RadioML2016.10a dataset, CDSCNN outperforms its counterparts by 1%–11%. After finetuning on the RadioML2016.10b dataset, the OA reaches 63.15%, demonstrating the robust recognition and generalization capability of CDSCNN. Moreover, the CDSCNN exhibits lower model complexity compared to other methods.

## Datasets

We conducted experiments on two datasets, namely RadioML2016.10a, and RadioML2016.10b.

| dataset     | modulation formats                                           | samples              |
| ----------- | ------------------------------------------------------------ | -------------------- |
| RadioML2016.10a | 8 digital formats: 8PSK, BPSK, CPFSK, GFSK, PAM4, 16QAM, 64QAM, QPSK; 3 analog formats: AM-DSB, AM-SSB, WBFM | 220 thousand (2×128) |
| RadioML2016.10b | 8 digital formats: 8PSK, BPSK, CPFSK, GFSK, PAM4, 16QAM, 64QAM, QPSK; 2 analog formats: AM-DSB, WBFM | 1.2 million (2×128)  |

## Requirements

- python == 3.10.4
- pytorch == 1.12.0
- scikit-learn == 1.3.0
- numpy == 1.21.5

## Citation

Please consider citing our paper if you find it helpful in your research. Please do not hesitate to contact us *(Email: ch_xiao@stu.xidian.edu.cn)* if there are any problems. 

```
@ARTICLE{10198896,
	author={Xiao, Chenghong and Yang, Shuyuan and Feng, Zhixi},
	journal={IEEE Transactions on Instrumentation and Measurement}, 
	title={Complex-Valued Depthwise Separable Convolutional Neural Network for Automatic Modulation Classification}, 
	year={2023},
	volume={72},
	pages={1-10},
	doi={10.1109/TIM.2023.3298657}
}
```
