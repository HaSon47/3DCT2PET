## Cách sử dụng

**Bước 1: Chỉnh sửa file `3DCT2PET_func.py`**

Mở file `3DCT2PET_func.py` và tìm đến khối `if __name__ == '__main__':`. Chỉnh sửa 3 đường dẫn sau cho phù hợp với máy của bạn:

```python
if __name__ == '__main__':
    CONFIG_FILE_PATH = '/home/PET-CT/huutien/Reg-GAN/Yaml/CycleGan.yaml'
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

