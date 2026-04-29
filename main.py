import struct
import argparse
import json
from pathlib import Path

# 基本类型大小
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

USR_MAGIC = 0x52535553
RSZ_MAGIC = 0x5A5352


def _align(pos: int, align: int) -> int:
    return (pos + align - 1) & ~(align - 1)


class FieldDef:
    def __init__(self, data: dict):
        self.name = data.get("name", "")
        self.type = data.get("type", "")
        self.size = data.get("size", 0)
        if self.size <= 0:
            self.size = _SCALAR_SIZES.get(self.type, 4)
        self.align = data.get("align", 1)
        self.is_array = data.get("array", False)


class ClassDef:
    def __init__(self, name: str, crc: str):
        self.name = name
        self.crc = crc
        self.fields: list[FieldDef] = []


class TypeDB:
    def __init__(self, rsz_path: Path):
        self.classes: dict[int, ClassDef] = {}
        if not rsz_path.exists():
            print(f"[WARN] RSZ template not found: {rsz_path}")
            return

        with rsz_path.open("r", encoding="utf-8-sig") as f:
            db = json.load(f)

        for hash_str, cdata in db.items():
            name = cdata.get("name", "")
            if not name:
                continue

            # The key is murmur3 hash in hex (e.g. "fa7ba5f9")
            h_int = int(hash_str, 16)
            cdef = ClassDef(name, cdata.get("crc", "0"))

            fields = []
            for fdata in cdata.get("fields", []):
                fields.append(FieldDef(fdata))
            cdef.fields = fields

            self.classes[h_int] = cdef

    def get_class(self, h: int) -> ClassDef | None:
        return self.classes.get(h)


class User3Parser:
    def __init__(self, data: bytes, typedb: TypeDB):
        self.buf = bytearray(data)
        self.typedb = typedb
        self.instance_starts: dict[int, int] = {}
        self.instance_classes: dict[int, ClassDef] = {}
        self._diag_limit = 20
        self._diag_printed = 0
        self._diag_oob_count = 0
        self._diag_suspicious_count = 0

    def _diag(self, msg: str, kind: str) -> None:
        if kind == "oob":
            self._diag_oob_count += 1
        elif kind == "suspicious":
            self._diag_suspicious_count += 1

        if self._diag_printed < self._diag_limit:
            print(msg)
            self._diag_printed += 1

    def _u32(self, pos: int) -> int:
        return struct.unpack_from("<I", self.buf, pos)[0]

    def _i32(self, pos: int) -> int:
        return struct.unpack_from("<i", self.buf, pos)[0]

    def _u64(self, pos: int) -> int:
        return struct.unpack_from("<Q", self.buf, pos)[0]

    def _i64(self, pos: int) -> int:
        return struct.unpack_from("<q", self.buf, pos)[0]

    def parse(self) -> bool:
        usr_magic = self._u32(0)
        # Often user.3 files start with 'USR\x00' (0x00525355)
        if usr_magic not in (0x00525355, 0x52535553):
            print(f"Not a valid user file. Found magic: {usr_magic:#010x}")
            return False
        rs = self._u64(32)
        rsz_magic = self._u32(rs)
        # RSZ magic is usually 'RSZ\x00' (0x005a5352)
        if rsz_magic not in (0x005A5352, 0x5A5352, 0x5A5352):
            print(f"Not a valid RSZ block. Found magic: {rsz_magic:#010x}")
            return False

        inst_count = self._i32(rs + 12)
        inst_off = self._i64(rs + 24)
        dat_off = self._i64(rs + 32)

        inst_hashes: list[int] = []
        for i in range(inst_count):
            inst_hashes.append(self._u32(rs + inst_off + i * 8))

        cursor = rs + dat_off
        for idx, h in enumerate(inst_hashes):
            if idx == 0:
                continue
            cls = self.typedb.get_class(h)
            if cls is None:
                print(f"[WARN] Missing class for hash {h:#010x} at instance {idx}")
                continue

            self.instance_classes[idx] = cls
            if not cls.fields:
                self.instance_starts[idx] = cursor
                continue

            first = cls.fields[0]
            cursor = _align(cursor, 4 if first.is_array else max(first.align, 1))
            self.instance_starts[idx] = cursor
            cursor = self._parse_instance(cursor, cls)

        suppressed = (self._diag_oob_count + self._diag_suspicious_count) - self._diag_printed
        if self._diag_oob_count or self._diag_suspicious_count:
            print(
                "[DIAG] Summary:"
                f" suspicious={self._diag_suspicious_count},"
                f" oob={self._diag_oob_count},"
                f" printed={self._diag_printed}"
                + (f", suppressed={suppressed}" if suppressed > 0 else "")
            )

        return True

    def _parse_instance(self, cursor: int, cls: ClassDef) -> int:
        buf_size = len(self.buf)
        for fld in cls.fields:
            if fld.is_array:
                cursor = _align(cursor, 4)
                if cursor + 4 > buf_size:
                    self._diag(
                        f"  [DIAG] OOB reading array count: class={cls.name} field={fld.name} cursor={cursor:#x} buf={buf_size:#x}",
                        "oob",
                    )
                    return cursor
                count = self._i32(cursor)
                if count < 0 or count > 100000:
                    raw = list(self.buf[cursor:cursor+4])
                    self._diag(
                        f"  [DIAG] Suspicious array count={count} at cursor={cursor:#x}: bytes={raw} class={cls.name} field={fld.name}",
                        "suspicious",
                    )
                cursor += 4
                if count > 0:
                    cursor = _align(cursor, max(fld.align, 1))
                    cursor += count * fld.size
            else:
                cursor = _align(cursor, max(fld.align, 1))
                cursor += fld.size
        return cursor


def _is_reasonable_i32(v: int) -> bool:
    return -1_000_000_000 <= v <= 1_000_000_000


def _is_reasonable_f32(v: float) -> bool:
    return (-1.0e9 < v < 1.0e9) and (v == v)


def _fallback_patch_condition_unit(
    buf: bytearray, target_hash: int, patch_value: float
) -> tuple[int, list[int], list[tuple[int, float, float]]]:
    """
    Fallback path:
    scan for ConditionUnitBase signature directly in binary and patch LimitValueF (+8).
    """
    pat = struct.pack("<I", target_hash)
    patch_count = 0
    patched_offsets: list[int] = []
    patched_samples: list[tuple[int, float, float]] = []
    pos = 0
    buf_len = len(buf)

    while True:
        pos = buf.find(pat, pos)
        if pos < 0:
            break

        # ConditionUnitBase layout starts with:
        # Hash(U32) + LimitValue(S32) + LimitValueF(F32) + TriggerNameHash(U32) + ...
        if pos + 36 <= buf_len:
            limit_i = struct.unpack_from("<i", buf, pos + 4)[0]
            limit_f = struct.unpack_from("<f", buf, pos + 8)[0]
            prop_count = struct.unpack_from("<i", buf, pos + 32)[0]

            # Heuristic guard to avoid patching unrelated hash occurrences.
            if (
                _is_reasonable_i32(limit_i)
                and _is_reasonable_f32(limit_f)
                and 0 <= prop_count <= 4096
            ):
                before = limit_f
                struct.pack_into("<f", buf, pos + 8, patch_value)
                patch_count += 1
                patched_offsets.append(pos)
                patched_samples.append((pos, before, patch_value))

        pos += 4

    return patch_count, patched_offsets, patched_samples


def build_default_output_path(input_file: Path) -> Path:
    """
    Keep natives directory nesting under data/output.
    Example:
      data/user3/natives/stm/singletonuserdata/missionuserdata.user.3
      -> data/output/natives/stm/singletonuserdata/missionuserdata.user.3
    """
    parts = list(input_file.parts)
    if "natives" in parts:
        idx = parts.index("natives")
        rel = Path(*parts[idx:])
        return Path("data/output") / rel
    return Path("data/output") / input_file.name


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

    print(f"Loading TypeDB from {rsz_path}...")
    typedb = TypeDB(rsz_path)
    if not typedb.classes:
        return False

    print(f"Reading {input_file}...")
    try:
        data = input_file.read_bytes()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return False

    parser = User3Parser(data, typedb)
    print("Parsing RSZ structure...")
    if not parser.parse():
        return False

    patch_count = 0
    # Now we iterate through correctly parsed instances!
    for idx, start_pos in parser.instance_starts.items():
        cls = parser.instance_classes.get(idx)
        if not cls:
            continue

        # WE ONLY PATCH ConditionUnit! ConditionMenuItem WILL BE IGNORED!
        if cls.name == "app.MissionUserdata.Unit.ConditionUnit":
            # Verification: we know the structure has Hash at +0 and LimitValueF at +8 (since it inherits ConditionUnitBase)
            try:
                actual_hash = struct.unpack_from("<I", parser.buf, start_pos)[0]
                limit_val_f = struct.unpack_from("<f", parser.buf, start_pos + 8)[0]
            except struct.error:
                continue

            if actual_hash == target_hash:
                print(
                    f"Found ConditionUnit [{idx}] with target Hash {target_hash} at offset {start_pos:#x}"
                )
                print(f"  -> Original LimitValueF = {limit_val_f:.1f}")

                struct.pack_into("<f", parser.buf, start_pos + 8, patch_value)
                print(f"  [+] Patched LimitValueF to {patch_value}!")
                patch_count += 1

    if patch_count == 0:
        print("Primary parser found no matches, trying binary-signature fallback...")
        fallback_count, fallback_offsets, fallback_samples = _fallback_patch_condition_unit(
            parser.buf, target_hash, patch_value
        )
        patch_count += fallback_count
        if fallback_count > 0:
            print(
                f"  [+] Fallback patched {fallback_count} entries "
                f"(first offsets: {[hex(x) for x in fallback_offsets[:5]]})"
            )
            for off, before, after in fallback_samples:
                print(
                    f"      - offset={off:#x}: LimitValueF {before:.1f} -> {after:.1f}"
                )

    print(f"\nTotal patched: {patch_count}")
    if patch_count > 0:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(parser.buf)
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
        required=False,
        default=None,
        help="Path to output missionuserdata.user.3 file (default: data/output with natives nesting)",
    )
    parser.add_argument(
        "--rsz",
        type=str,
        default="data/rszpragmata.json",
        help="Path to RSZ template json",
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
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else build_default_output_path(input_path)
    patch_mission_user_data(input_path, output_path, Path(args.rsz), args.hash, args.limit)


if __name__ == "__main__":
    main()
