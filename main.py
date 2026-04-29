import struct
import argparse
import json
from pathlib import Path

# 基本类型大小，用于辅助计算偏移
_SCALAR_SIZES: dict[str, int] = {
    "Bool": 1,
    "S8": 1,
    "U8": 1,
    "S16": 2,
    "U16": 2,
    "S32": 4,
    "U32": 4,
    "F32": 4,
    "Enum": 4,
    "Sfix": 4,
    "S64": 8,
    "U64": 8,
    "F64": 8,
    "Object": 4,
    "UserData": 4,
    "Guid": 16,
    "GameObjectRef": 16,
    "Uri": 16,
    "Float2": 8,
    "Vec2": 8,
    "Float3": 16,
    "Vec3": 16,
    "Position": 16,
    "Float4": 16,
    "Vec4": 16,
    "Quaternion": 16,
    "Color": 16,
    "AABB": 24,
    "Capsule": 32,
    "OBB": 52,
    "Mat3": 36,
    "Mat4": 64,
}


def get_relative_offset_from_rsz(rsz_path: Path) -> int | None:
    """
    通过解析 rszpragmata.json (TypeDB)，自动寻找定义了 Hash 和 LimitValueF 的类，
    并计算出 LimitValueF 相对 Hash 的准确字节偏移量。
    这能保证游戏更新后如果类结构发生变化（比如中间插入了新字段），我们也能自适应。
    """
    if not rsz_path.exists():
        return None

    try:
        with rsz_path.open("r", encoding="utf-8-sig") as f:
            db = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse RSZ template: {e}")
        return None

    # 寻找同时包含 Hash 和 LimitValueF 的类 (ConditionUnitBase 或 ConditionUnit)
    for cls_data in db.values():
        fields = cls_data.get("fields", [])
        hash_idx = -1
        limit_idx = -1

        for i, f in enumerate(fields):
            fname = f.get("name")
            if fname == "Hash":
                hash_idx = i
            elif fname == "LimitValueF":
                limit_idx = i

        if hash_idx != -1 and limit_idx != -1:
            # 找到目标类，现在模拟内存对齐计算相对偏移
            cursor = 0
            hash_offset = -1
            limit_offset = -1

            for i, f in enumerate(fields):
                align = max(int(f.get("align", 1)), 1)
                if f.get("array", False):
                    align = 4

                cursor = (cursor + align - 1) & ~(align - 1)

                if i == hash_idx:
                    hash_offset = cursor
                elif i == limit_idx:
                    limit_offset = cursor

                size = int(f.get("size", 0))
                if size <= 0:
                    t = f.get("type", "")
                    size = _SCALAR_SIZES.get(t, 4)  # 默认使用4字节
                cursor += size

            relative_offset = limit_offset - hash_offset
            print(
                f"[INFO] Auto-detected offset from RSZ: LimitValueF is {relative_offset} bytes after Hash."
            )
            return relative_offset

    print(
        "[WARN] Could not find Hash and LimitValueF in the same class in RSZ template."
    )
    return None


def patch_mission_user_data(
    input_file: Path,
    output_file: Path,
    rsz_path: Path,
    target_hash: int = 3410781912,
    patch_value: float = 99999.0,
) -> bool:
    if not input_file.exists():
        print(f"[ERROR] File not found: {input_file}")
        return False

    print(f"Reading {input_file}...")
    try:
        buf = bytearray(input_file.read_bytes())
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return False

    # 优先尝试从 RSZ 模板获取正确的相对偏移量，保证未来兼容性
    relative_offset = get_relative_offset_from_rsz(rsz_path)
    if relative_offset is None:
        print("[INFO] Using default relative offset (+8 bytes).")
        relative_offset = 8

    hash_bytes = struct.pack("<I", target_hash)
    patch_count = 0
    offset = 0

    while True:
        offset = buf.find(hash_bytes, offset)
        if offset == -1:
            break

        try:
            # 读取 Hash 之后偏移位置的 LimitValueF 原值
            limit_val_f = struct.unpack_from("<f", buf, offset + relative_offset)[0]
        except struct.error:
            offset += 4
            continue

        print(f"Found target Hash {target_hash} at offset {offset:#x}")
        print(f"  -> Original LimitValueF = {limit_val_f:.1f}")

        struct.pack_into("<f", buf, offset + relative_offset, patch_value)
        print(f"  [+] Patched LimitValueF to {patch_value}!")
        patch_count += 1

        offset += 4

    print(f"\nTotal patched: {patch_count}")
    if patch_count > 0:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(buf)
            print(f"Saved modified file to {output_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to write file: {e}")
            return False
    else:
        print("No matches found, nothing saved.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Patch missionuserdata.user.3 LimitValueF based on Hash."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input missionuserdata.user.3 file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output missionuserdata.user.3 file",
    )
    parser.add_argument(
        "--rsz",
        type=str,
        default="data/rszpragmata.json",
        help="Path to RSZ template json (for offset compatibility)",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=3410781912,
        help="Target Hash to search for (default: 3410781912)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=99999.0,
        help="New LimitValueF float value (default: 99999.0)",
    )

    args = parser.parse_args()

    patch_mission_user_data(
        Path(args.input), Path(args.output), Path(args.rsz), args.hash, args.limit
    )


if __name__ == "__main__":
    main()
