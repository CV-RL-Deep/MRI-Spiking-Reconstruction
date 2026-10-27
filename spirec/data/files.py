import glob
import os
import re
from typing import Dict, List, Optional

from spirec.core.utils import logger

EXCLUDE_FILES = {
    "wmIXI338-HH-1971-MADisoTFE1_-s3T188_-0301-00003-000001-01.nii",
    "wmIXI357-HH-2076-MADisoTFE1_-s3T199_-0301-00003-000001-01.nii",
    "wmIXI402-Guys-0961-MPRAGESEN_-s413_-0301-00003-000001-01.nii",
    "wmIXI423-IOP-0974-SAGFSPGR_-sIXI00_-0003-00001-000001-01.nii",
}


def get_subject_id_from_path(path: str) -> str:
    """Extracts standardized subject ID from path string."""
    filename = os.path.basename(path)
    match = re.search(r"IXI\d{3}", filename)
    if match:
        return match.group(0)
    return filename.replace(".nii.gz", "").replace(".nii", "")


def find_t1_files(root_dir: str) -> List[str]:
    """Recursively indexes valid T1 NIfTI volumes, filtering corrupted files."""
    pattern = os.path.join(root_dir, "**", "wm*.nii*")
    matches = glob.glob(pattern, recursive=True)

    valid_files = [
        path
        for path in matches
        if os.path.getsize(path) > 1024 and os.path.basename(path) not in EXCLUDE_FILES
    ]
    logger.info(f"Indexed {len(valid_files)} valid T1 volumes in: {root_dir}")
    return sorted(valid_files)


def get_brats_subjects(root_dir: str) -> List[Dict[str, Optional[str]]]:
    """Indexes paired T1 and Segmentation NIfTI volumes from BraTS directories."""
    subjects = []
    t1_pattern = os.path.join(root_dir, "**", "*_t1.nii*")
    t1_files = glob.glob(t1_pattern, recursive=True)

    for t1_path in sorted(t1_files):
        if os.path.getsize(t1_path) <= 1024 or "t1ce" in t1_path.lower():
            continue

        folder = os.path.dirname(t1_path)
        subject_id = os.path.basename(t1_path).replace("_t1.nii.gz", "").replace("_t1.nii", "")

        seg_path = os.path.join(folder, f"{subject_id}_seg.nii.gz")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(folder, f"{subject_id}_seg.nii")
        if not os.path.exists(seg_path):
            alts = glob.glob(os.path.join(folder, "*seg*.nii*"))
            seg_path = alts[0] if alts else None

        subjects.append({"id": subject_id, "t1": t1_path, "seg": seg_path})

    logger.info(f"Indexed {len(subjects)} BraTS paired subjects in: {root_dir}")
    return subjects

