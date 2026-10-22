"""登记并显示实验身份信息，供阶段截图和报告验收使用。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


EXPERIMENT_ID = "04"
IDENTITY_PATH = (
    Path(__file__).resolve().parent / "outputs" / "student_identity.json"
)
STUDENT_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{4,32}$")


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_student_id(value: str) -> str:
    student_id = value.strip()
    if not STUDENT_ID_PATTERN.fullmatch(student_id):
        raise ValueError("学号需为 4～32 位字母、数字、连字符或下划线")
    return student_id


def _identity_code(
    student_id: str, experiment_id: str, registered_at: str
) -> str:
    raw = f"{student_id}|{experiment_id}|{registered_at}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12].upper()


def register_identity(student_id: str) -> dict[str, str]:
    student_id = _normalize_student_id(student_id)
    registered_at = _now_text()
    record = {
        "student_id": student_id,
        "experiment_id": EXPERIMENT_ID,
        "registered_at": registered_at,
        "identity_code": _identity_code(
            student_id, EXPERIMENT_ID, registered_at
        ),
    }
    IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDENTITY_PATH.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def load_identity() -> dict[str, str]:
    if not IDENTITY_PATH.exists():
        raise FileNotFoundError("尚未登记学号，请先使用 --set")
    record = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))
    student_id = _normalize_student_id(str(record["student_id"]))
    registered_at = str(record["registered_at"])
    if str(record["experiment_id"]) != EXPERIMENT_ID:
        raise ValueError("身份文件中的实验编号与当前实验不一致")
    expected = _identity_code(student_id, EXPERIMENT_ID, registered_at)
    if str(record["identity_code"]) != expected:
        raise ValueError("身份文件校验失败，请重新登记学号")
    return record


def display_identity(record: dict[str, str], stage: str) -> None:
    print(f"STUDENT_ID={record['student_id']}")
    print(f"EXPERIMENT_ID={record['experiment_id']}")
    print(f"STAGE={stage.strip() or 'checkpoint'}")
    print(f"IDENTITY_CODE={record['identity_code']}")
    print(f"REGISTERED_AT={record['registered_at']}")
    print(f"CAPTURED_AT={_now_text()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验身份检查点")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--set", dest="student_id", metavar="STUDENT_ID")
    mode.add_argument("--show", action="store_true")
    parser.add_argument("--stage", default="checkpoint")
    args = parser.parse_args()
    record = (
        register_identity(args.student_id)
        if args.student_id is not None
        else load_identity()
    )
    display_identity(record, args.stage)


if __name__ == "__main__":
    main()
