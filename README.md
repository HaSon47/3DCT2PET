# Breaking the Dilemma of Medical Image-to-image Translation
3DCT2PET
## Main Reference Environment
1. Linux         (Titan RTX)
2. Python        (3.6.6)
3. torch         (1.9.0+cu111)
5. visdom        (0.1.8.9)
6. numpy         (1.19.2)
7. skimage       (0.15.0)
8. Yaml          (5.4.1)
9. cv2           (3.4.2)
10. PIL          (8.3.2)

## Environmen setup
To create the environment using `conda`, run:
```bash
conda env create -f environment.yml
```

## Checkpoint
Thay đường dẫn tới checkpoint của bạn trong file `Yaml/CycleGan.yaml'
```python
checkpoint_path: 'path/to/checkpoint'
```
## Usage

### Using as an API 
#### 3D CT Numpy to 3D PET Numpy
```python
from 3DCT2PET_func import infer_ct_to_pet
```
- input: 3D CT numpy array
- output: 3D PET numpy array

#### 3D PET Numpy to PET DICOM
```python
from 3DCT2PET_func import convert_ct_to_pet
```
- input:
```python
"""
    pet_npy_path: path to your 3D Pet npy 
    ct_dcm_folder: path to coresponding CT dicom of CT npy
    pet_template_path: path to template of a Pet dicom file
    output_folder: path to save output Pet dicom folder
"""
```


### Using via Terminal
**Bước 1: Chỉnh sửa file `3DCT2PET_func.py`**

Mở file `3DCT2PET_func.py` và tìm đến khối `if __name__ == '__main__':`. Chỉnh sửa 2 đường dẫn sau cho phù hợp với máy của bạn:

```python
if __name__ == '__main__':
    # ... 
    INPUT_CT_NPY_PATH = 'path/to/your/input_ct_file.npy'  # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY
    OUTPUT_PET_NPY_PATH = 'output/result_pet.npy'        # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY
    
    # ... phần còn lại của code
```

**Bước 2: Chạy script**

Mở terminal và chạy lệnh:

```bash
python 3DCT2PET_func.py
```

**Bước 3: Kiểm tra kết quả**

Sau khi script chạy xong, một file `.npy` chứa kết quả PET 3D sẽ được tạo ra tại đường dẫn `OUTPUT_PET_NPY_PATH` mà bạn đã chỉ định.




## Citation

If you find RegGAN useful in your research, please consider citing:

```
@inproceedings{
kong2021breaking,
title={Breaking the Dilemma of Medical Image-to-image Translation},
author={Lingke Kong and Chenyu Lian and Detian Huang and ZhenJiang Li and Yanle Hu and Qichao Zhou},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=C0GmZH2RnVR}
}
```
