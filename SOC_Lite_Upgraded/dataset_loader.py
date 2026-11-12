"""Readers for the *original* NSL-KDD and CICIDS2017 datasets.

No benchmark samples are bundled with this application. The user supplies official
files, and dataset provenance is kept distinct from the built-in synthetic demo.
"""
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

Source = Union[str, Path, object]

# Official NSL-KDD TXT layout: 41 features, attack name, difficulty score.
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Known sub-attack types of NSL-KDD. The literature occasionally differs on
# worm/httptunnel; this mapping groups worm as R2L and httptunnel as U2R.
NSL_KDD_ATTACK_GROUPS = {
    "DoS": ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
            "mailbomb", "processtable", "udpstorm"],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
            "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
            "snmpguess", "xlock", "xsnoop", "worm"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "rootkit",
            "httptunnel", "ps", "sqlattack", "xterm"],
}
_ATTACK_TO_GROUP = {attack: group for group, names in NSL_KDD_ATTACK_GROUPS.items()
                    for attack in names}

# Metadata should not become predictive features (identifiers can cause leakage).
CICIDS2017_DROP_COLUMNS = {
    "flow id", "source ip", "src ip", "destination ip", "dst ip",
    "timestamp", "unnamed: 0", "attack category", "attack type",
    "binary label", "binary_label", "class", "attempted category",
}


def read_nsl_kdd(source: Source) -> pd.DataFrame:
    """Read headerless KDDTrain+.txt/KDDTest+.txt; map attack names to 5 groups."""
    if hasattr(source, "seek"):
        source.seek(0)
    raw = pd.read_csv(source, header=None, low_memory=False)
    if raw.empty:
        raise ValueError("File NSL-KDD rỗng.")
    # Tolerate an optional header row in an exported CSV.
    if str(raw.iloc[0, 0]).strip().lower() == "duration":
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] not in (42, 43):
        raise ValueError(
            f"NSL-KDD gốc phải có 42 hoặc 43 cột (41 đặc trưng, nhãn, "
            f"và tùy chọn difficulty); nhận được {raw.shape[1]} cột."
        )
    raw.columns = NSL_KDD_FEATURES + ["attack_name"] + (
        ["difficulty"] if raw.shape[1] == 43 else []
    )
    names = (raw["attack_name"].astype("string").str.strip()
             .str.lower().str.rstrip("."))
    labels = names.map({"normal": "Normal", **_ATTACK_TO_GROUP})
    unknown = sorted(names[labels.isna()].dropna().unique().tolist())
    if unknown:
        raise ValueError(f"NSL-KDD có nhãn tấn công chưa được ánh xạ: {unknown[:12]}")
    if labels.isna().any():
        raise ValueError("NSL-KDD chứa nhãn trống, vui lòng kiểm tra nguồn dữ liệu.")
    result = raw[NSL_KDD_FEATURES].copy()
    for col in NSL_KDD_FEATURES:
        if col not in ("protocol_type", "service", "flag"):
            result[col] = pd.to_numeric(result[col], errors="coerce")
        else:
            result[col] = result[col].astype("string").str.strip().astype(object)
    result["label"] = labels.astype(str).to_numpy()
    return result


def _clean_cic_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk.columns = [str(c).lstrip("\ufeff").strip() for c in chunk.columns]
    if chunk.columns.duplicated().any():
        raise ValueError("CICIDS2017 có cột trùng tên sau khi loại khoảng trắng.")
    label_columns = [c for c in chunk.columns if c.casefold() == "label"]
    if len(label_columns) != 1:
        raise ValueError("CICIDS2017 phải có cột 'Label' chứa nhãn lưu lượng.")
    chunk.rename(columns={label_columns[0]: "label"}, inplace=True)
    if not any(c.casefold() == "flow duration" for c in chunk.columns):
        raise ValueError("Thiếu cột 'Flow Duration': file không giống CICIDS2017 MachineLearningCSV.")
    to_drop = [c for c in chunk.columns if c.casefold() in CICIDS2017_DROP_COLUMNS]
    chunk.drop(columns=to_drop, inplace=True)
    chunk = chunk.dropna(subset=["label"])
    chunk["label"] = chunk["label"].astype(str).str.strip()
    chunk = chunk.loc[chunk["label"] != ""].copy()
    chunk["label"] = chunk["label"].replace({
        "BENIGN": "Normal", "Benign": "Normal", "benign": "Normal",
        "normal": "Normal", "NORMAL": "Normal"
    })
    return chunk


def _read_one_cic_file(source: Source, max_rows: int, seed: int) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    rng = np.random.default_rng(seed)
    sample = None
    # Each row receives an independent random priority. Keeping the smallest
    # priorities yields an approximately uniform reservoir across the entire file,
    # unlike taking only the first N rows of a day-specific attack file.
    reader = pd.read_csv(source, chunksize=10000, encoding="utf-8-sig", low_memory=False)
    for chunk in reader:
        chunk = _clean_cic_chunk(chunk)
        if chunk.empty:
            continue
        chunk["__sample_priority__"] = rng.random(len(chunk))
        if sample is None:
            sample = chunk
        else:
            if set(sample.columns) != set(chunk.columns):
                raise ValueError("Tên cột CICIDS2017 thay đổi giữa các phần của file.")
            sample = pd.concat([sample, chunk], ignore_index=True)
        if len(sample) > max_rows:
            sample = sample.nsmallest(max_rows, "__sample_priority__")
    if sample is None or sample.empty:
        raise ValueError("Không đọc được bản ghi hợp lệ trong file CICIDS2017.")
    return sample.drop(columns=["__sample_priority__"]).reset_index(drop=True)


def read_cicids2017(sources: Iterable[Source], max_rows_per_file: int = 20000,
                    random_state: int = 42) -> pd.DataFrame:
    """Read real CICIDS2017 MachineLearningCSV files; sample each file if large.

    Attack labels remain original CICIDS2017 labels (e.g. 'DoS Hulk'), rather
    than incorrectly reassigning them to NSL-KDD's R2L/U2R taxonomy.
    """
    sources = list(sources)
    if not sources:
        raise ValueError("Hãy chọn ít nhất một file CSV CICIDS2017.")
    if max_rows_per_file < 1:
        raise ValueError("max_rows_per_file phải >= 1.")
    frames = [_read_one_cic_file(source, max_rows_per_file, random_state + i)
              for i, source in enumerate(sources)]
    expected = set(frames[0].columns)
    for i, frame in enumerate(frames[1:], start=2):
        if set(frame.columns) != expected:
            raise ValueError(f"File CICIDS2017 thứ {i} có schema khác file thứ nhất. "
                             "Hãy dùng các file cùng phiên bản MachineLearningCSV.")
    data = pd.concat(frames, ignore_index=True)
    if data["label"].nunique() < 2:
        raise ValueError("Dữ liệu CICIDS2017 đã chọn chỉ có một nhãn. "
                         "Hãy nạp thêm file có lưu lượng Normal và ít nhất một loại tấn công.")
    for col in data.columns:
        if col == "label":
            continue
        # Convert 'Infinity', '-Infinity', mixed numeric text and floats to numeric.
        if not pd.api.types.is_numeric_dtype(data[col]):
            numeric = pd.to_numeric(data[col], errors="coerce")
            nonempty = data[col].notna().sum()
            if nonempty and numeric.notna().sum() / nonempty >= 0.85:
                data[col] = numeric
            else:
                data[col] = data[col].astype(object)
    return data
