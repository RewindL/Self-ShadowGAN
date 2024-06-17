## \[Self-ShadowGAN\] Learning to Remove Shadows from a Single Image
This is the PyTorch implementation of the paper "[Learning to Remove Shadows from a Single Image](https://link.springer.com/article/10.1007/s11263-023-01823-9)" published on International Journal of Computer Vision (IJCV).


## Requirements
- python (3.7 or newer version)
- pytorch (1.4 or newer version)
- tochvision

  

## Training & Inference
`python main.py --gpu gpu_idx --idx image_idx` (e.g. `python main.py --gpu 0 --idx 1`)



## Notes

- If you use the code, please cite our paper.

> ```
> @article{jiang2023learning,
>   title={Learning to Remove Shadows from a Single Image},
>   author={Jiang, Hao and Zhang, Qing and Nie, Yongwei and Zhu, Lei and Zheng, Wei-Shi},
>   journal={International Journal of Computer Vision},
>   volume={131},
>   number={9},
>   pages={2471--2488},
>   year={2023},
>   publisher={Springer}
> }
> ```

