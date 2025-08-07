# inference_api.py

import argparse
import os
import torch
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable # Cần thiết nếu trainer của bạn vẫn dùng
import copy
import pydicom
from pydicom.uid import generate_uid

from trainer import CycTrainerZoom 

# --- Cài đặt môi trường (giữ nguyên) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# --- Biến toàn cục để cache trainer, giúp tăng tốc API ---
initialized_trainers = {}

def get_config(config_path):
    with open(config_path, 'r') as stream:
        return yaml.safe_load(stream)

def get_trainer(config_path):
    """
    Khởi tạo hoặc lấy lại trainer đã được load.
    Tránh phải load lại model mỗi lần gọi, cực kỳ quan trọng cho hiệu năng.
    """
    if config_path in initialized_trainers:
        print("INFO: Returning cached trainer.")
        return initialized_trainers[config_path]
    
    print(f"INFO: Initializing new trainer for: {config_path}")
    config = get_config(config_path)
    
    if config['name'] == 'CycleGanZoom':
        trainer = CycTrainerZoom.Cyc_Trainer(config)
    else:
        raise ValueError(f"Unsupported trainer name: {config['name']}")
    # Lưu trainer vào cache để dùng lại lần sau
    initialized_trainers[config_path] = trainer
    return trainer


def infer_ct_to_pet(ct_numpy_array: np.ndarray, config_path: str = '/home/PET-CT/huutien/Reg-GAN/Yaml/CycleGan.yaml') -> np.ndarray:
    """
    Hàm API chính: Nhận mảng CT, trả về mảng PET.
    """
    # 1. Lấy model đã được load sẵn
    trainer = get_trainer(config_path)

    # 2. Gọi phương thức inference mà chúng ta đã xây dựng trong trainer
    print("INFO: Starting inference on the input volume...")
    pet_numpy_output = trainer.infer_from_numpy(ct_numpy_array)
    print("INFO: Inference finished.")

    return pet_numpy_output

def convert_ct_to_pet(
    pet_npy_path,
    ct__dcm_folder,
    pet_template_path,
    output_folder
):
    # Load predicted PET data
    pet_npy = np.load(pet_npy_path)

    # Hàm đọc file DICOM và lấy SliceLocation nếu có
    def dcm_to_array(dcm_path):
        dcm = pydicom.dcmread(dcm_path)
        return dcm.SliceLocation if 'SliceLocation' in dcm else None

    # Tạo danh sách các file CT kèm SliceLocation
    ct_pair = []
    for fname in os.listdir(ct__dcm_folder):
        dcm_path = os.path.join(ct__dcm_folder, fname)
        lo = dcm_to_array(dcm_path)
        ct_pair.append([fname, lo])
    ct_pair = sorted(ct_pair, key=lambda x: x[1] if x[1] is not None else 0)

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_folder, exist_ok=True)

    # Đọc PET mẫu
    pet_template = pydicom.dcmread(pet_template_path)

    # Bắt đầu xử lý từng lát cắt
    for ind, pair in enumerate(ct_pair):
        try:
            ct_path = os.path.join(ct__dcm_folder, pair[0])
            ct_dicom = pydicom.dcmread(ct_path)

            # Tạo bản sao PET
            pet_dicom = copy.deepcopy(pet_template)

            # Copy metadata từ CT
            pet_dicom.PatientID = ct_dicom.PatientID
            pet_dicom.PatientName = ct_dicom.PatientName
            pet_dicom.StudyInstanceUID = ct_dicom.StudyInstanceUID
            pet_dicom.SOPInstanceUID = generate_uid()

            if hasattr(pet_dicom, "file_meta"):
                pet_dicom.file_meta.MediaStorageSOPInstanceUID = generate_uid()

            pet_dicom.SeriesDescription = "Converted from CT to PET"
            pet_dicom.ReconstructionDiameter = ct_dicom.ReconstructionDiameter
            pet_dicom.FieldOfViewShape = getattr(ct_dicom, "FieldOfViewShape", "CIRCULAR")
            pet_dicom.FieldOfViewDimensions = getattr(ct_dicom, "FieldOfViewDimensions", [500, 500])
            pet_dicom.PatientPosition = ct_dicom.PatientPosition
            pet_dicom.ImagePositionPatient = ct_dicom.ImagePositionPatient
            pet_dicom.NumberOfSlices = pet_npy.shape[0]
            pet_dicom.ImageIndex = ind

            # Cập nhật pixel data
            pixel_data = pet_npy[ind]
            pixel_data = pixel_data.astype(np.uint16)
            pet_dicom.PixelData = pixel_data.tobytes()

            # Lưu file
            new_pet_path = os.path.join(output_folder, f"{ind+1:04d}.dcm")
            pet_dicom.save_as(new_pet_path)
            print(f"✅ Đã tạo {new_pet_path}")

        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý {pair[0]}: {e}")

    print("🔄 Hoàn tất chuyển đổi tất cả file CT sang PET.")
    print(f"📂 Kết quả đã được lưu tại: {output_folder}")



if __name__ == '__main__':
    # --- BƯỚC 1: CẤU HÌNH CÁC ĐƯỜNG DẪN ---
    # Bạn chỉ cần thay đổi 3 dòng này cho phù hợp với máy của bạn
    CONFIG_FILE_PATH = '/home/PET-CT/huutien/Reg-GAN/Yaml/CycleGan.yaml'
    INPUT_CT_NPY_PATH = '/workdir/radish/PET-CT/CT2PET_3d_npy/DICOM_000000006355_LƯỜNG THỊ MAI 17909/0002.npy' # Ví dụ
    OUTPUT_PET_NPY_PATH = '/home/PET-CT/huutien/Reg-GAN/output_demo_dienbien/DICOM_000000006355_LƯỜNG THỊ MAI 17909/output_pet_from_api.npy'
    
    # --- BƯỚC 2: TẢI DỮ LIỆU ĐẦU VÀO ---
    print(f"INFO: Loading input CT from: {INPUT_CT_NPY_PATH}")
    try:
        ct_input_array = np.load(INPUT_CT_NPY_PATH, allow_pickle=True)
        print(f"INFO: Input CT shape: {ct_input_array.shape}, dtype: {ct_input_array.dtype}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_CT_NPY_PATH}'. Please check the path.")
        exit() # Thoát chương trình nếu không tìm thấy file

    # --- BƯỚC 3: GỌI HÀM INFERENCE CHÍNH ---
    output_pet_array = infer_ct_to_pet(
        ct_numpy_array=ct_input_array,
        config_path=CONFIG_FILE_PATH
    )

    # --- BƯỚC 4: LƯU KẾT QUẢ ---
    print(f"INFO: Saving output PET to: {OUTPUT_PET_NPY_PATH}")
    np.save(OUTPUT_PET_NPY_PATH, output_pet_array)
    print(f"SUCCESS: Result saved. Output PET shape: {output_pet_array.shape}")