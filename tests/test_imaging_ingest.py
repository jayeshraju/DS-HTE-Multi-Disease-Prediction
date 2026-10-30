"""
Unit tests for Medical Image Ingestion and Validation Engine.
Follows Section 4.1:
- Valid zips and folder structures
- Zip-slip security prevention
- Corrupt files & bad magic bytes rejection
- Perceptual difference hash duplicate detection
- DICOM PHI de-identification
- Class threshold rules (warn <50, block <10)
"""

import os
import io
import shutil
import zipfile
import pytest
import numpy as np
from PIL import Image

from src.imaging.ingest import (
    MedicalImageIngestor, is_safe_zip_path, check_magic_bytes,
    compute_dhash, hamming_distance, deidentify_and_convert_dicom,
    HAS_PYDICOM
)


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    d = tmp_path / "test_dataset"
    d.mkdir()
    return str(d)


def test_zip_slip_prevention():
    assert is_safe_zip_path("C:/safe/dir", "image.png")
    assert is_safe_zip_path("C:/safe/dir", "subdir/image.png")
    assert not is_safe_zip_path("C:/safe/dir", "../evil.png")
    assert not is_safe_zip_path("C:/safe/dir", "../../windows/system32/cmd.exe")


def test_magic_bytes_detection():
    # PNG header
    png_bytes = b'\x89PNG\r\n\x1a\n' + b'\x00' * 20
    assert check_magic_bytes(png_bytes)

    # JPEG header
    jpg_bytes = b'\xff\xd8\xff\xe0' + b'\x00' * 20
    assert check_magic_bytes(jpg_bytes)

    # Corrupt / random bytes
    random_bytes = b'NOT_AN_IMAGE_HEADER_123456789'
    assert not check_magic_bytes(random_bytes)


def test_dhash_and_near_duplicates():
    # Generate an image and a slightly altered duplicate
    arr = np.random.RandomState(42).randint(0, 255, (64, 64), dtype=np.uint8)
    im1 = Image.fromarray(arr)
    # Exact duplicate
    im2 = Image.fromarray(arr)
    # Near duplicate (small noise)
    arr_near = np.clip(arr.astype(int) + np.random.RandomState(101).randint(-5, 5, (64, 64)), 0, 255).astype(np.uint8)
    im3 = Image.fromarray(arr_near)
    # Completely different image
    im_diff = Image.fromarray(np.random.RandomState(999).randint(0, 255, (64, 64), dtype=np.uint8))

    h1 = compute_dhash(im1)
    h2 = compute_dhash(im2)
    h3 = compute_dhash(im3)
    h_diff = compute_dhash(im_diff)

    assert hamming_distance(h1, h2) == 0  # Exact match
    assert hamming_distance(h1, h3) <= 4   # Near duplicate
    assert hamming_distance(h1, h_diff) > 4 # Non-duplicate


def test_ingestor_validation_report_blocking_and_warnings(tmp_dataset_dir):
    ingestor = MedicalImageIngestor(storage_dir=tmp_dataset_dir)
    
    # 1. Create a tiny dataset: Class A has 5 images (< 10 -> BLOCK), Class B has 20 images (< 50 -> WARN)
    class_a_dir = os.path.join(tmp_dataset_dir, "ClassA")
    class_b_dir = os.path.join(tmp_dataset_dir, "ClassB")
    os.makedirs(class_a_dir, exist_ok=True)
    os.makedirs(class_b_dir, exist_ok=True)

    for i in range(5):
        im = Image.new("RGB", (64, 64), color=(i * 10, i * 20, i * 30))
        im.save(os.path.join(class_a_dir, f"img_a_{i}.png"))

    for i in range(20):
        im = Image.new("RGB", (64, 64), color=(i * 5, i * 10, i * 15))
        im.save(os.path.join(class_b_dir, f"img_b_{i}.png"))

    report = ingestor.validate_dataset(tmp_dataset_dir)

    assert report["total_valid_images"] == 25
    assert report["num_classes"] == 2
    assert not report["can_train"]  # Blocked because ClassA < 10
    assert any("minimum 10 required" in msg for msg in report["blocking_issues"])
    assert any("fewer than 50 recommended" in msg for msg in report["warnings"])
    assert not report["has_patient_id"]


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom required for DICOM test")
def test_dicom_phi_deidentification(tmp_path):
    import pydicom
    from pydicom.dataset import FileMetaDataset, FileDataset

    # Construct a synthetic DICOM file with sensitive PHI attributes
    filename = str(tmp_path / "patient_scan.dcm")
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.UID("1.2.3")
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "PHI-SECRET-9988"
    ds.PatientBirthDate = "19600101"
    ds.InstitutionName = "Confidential Memorial Hospital"
    ds.Rows = 32
    ds.Columns = 32
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((32, 32), dtype=np.uint16).tobytes()
    ds.save_as(filename)

    # De-identify and convert to PNG
    out_png = str(tmp_path / "deidentified_slice.png")
    res = deidentify_and_convert_dicom(filename, out_png)

    assert res["status"] == "deidentified"
    assert "PatientName" in res["stripped_tags"]
    assert "PatientID" in res["stripped_tags"]
    assert os.path.exists(out_png)
