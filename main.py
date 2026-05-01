"""
基于 RSZ 模板推导 user.3 的二进制布局，并按"地址"直接修改字段。

核心思路：
  1. 用 rszpragmata.json 提供的字段元数据，从 user.3 二进制中按顺序解析每个 instance；
  2. 解析过程中为每个原子字段记录它在二进制文件中的偏移（"地址"），形成一棵
     与 dump JSON 形态接近、但每个值都自带 offset 的结构树；
  3. 修改时不再做任何"二次扫描 / 二次猜测"——直接用之前记录下来的 offset，把新值
     pack 回 buf 的对应位置。

支持的关键点：
  - RSZUserData 表：被 userdata 引用的 instance 在主数据流里占 0 字节，
    它们的"内容"是外部资源 path（保存在 userdata 表的字符串区）。
  - Object / UserData 字段在主数据流里只占 4 字节 ref index，被引用 instance 自己
    在 instance 列表中独占一段连续字段空间。
  - 数组：4 字节 count + 对齐 + count 个元素，元素按其类型读取。
  - String：4 字节 char count + UTF-16 数据（含结尾 NUL）。

调试输出（.temp/.../xxx.debug.json）：
  结构与 data/json/... 的 dump JSON 完全对齐（层次化、Object ref 内联展开），
  但每个字段值改为 {"__v": <值>, "__o": "<hex 偏移>"} 以便直接定位二进制位置。
  数组改为 {"__count": N, "__count_off": "<hex>", "elements": [...]}.
"""

from __future__ import annotations

import argparse
import json
import struct
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    "Resource": 4,
    "RuntimeType": 4,
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
    "OBB": 80,
    "Mat3": 36,
    "Mat4": 64,
    "Range": 8,
    "RangeI": 8,
    "Sphere": 16,
}

_VARLEN_STRING_TYPES = {"String", "ResourceFile", "RuntimeType"}


def _align(pos: int, align: int) -> int:
    align = max(int(align), 1)
    return (pos + align - 1) & ~(align - 1)


# ---------- 类型库 ----------


@dataclass
class FieldDef:
    name: str
    type: str
    size: int
    align: int
    is_array: bool
    original_type: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "FieldDef":
        ftype = data.get("type", "")
        size = int(data.get("size", 0))
        if size <= 0:
            size = _SCALAR_SIZES.get(ftype, 4)
        return cls(
            name=data.get("name", ""),
            type=ftype,
            size=size,
            align=int(data.get("align", 1) or 1),
            is_array=bool(data.get("array", False)),
            original_type=data.get("original_type", ""),
        )


@dataclass
class ClassDef:
    name: str
    crc: str
    fields: list[FieldDef] = field(default_factory=list)


class TypeDB:
    def __init__(self, rsz_path: Path):
        self.classes: dict[int, ClassDef] = {}
        if not rsz_path.exists():
            print(f"[ERROR] RSZ template not found: {rsz_path}")
            return

        with rsz_path.open("r", encoding="utf-8-sig") as f:
            db = json.load(f)

        for hash_str, cdata in db.items():
            name = cdata.get("name", "")
            if not name:
                continue
            try:
                h_int = int(hash_str, 16)
            except ValueError:
                continue
            cdef = ClassDef(name=name, crc=str(cdata.get("crc", "0")))
            for fdata in cdata.get("fields", []) or []:
                cdef.fields.append(FieldDef.from_json(fdata))
            self.classes[h_int] = cdef

    def get(self, type_hash: int) -> ClassDef | None:
        return self.classes.get(type_hash)


# ---------- RSZ 数据模型 ----------


@dataclass
class FieldNode:
    """解析后的字段节点，带二进制偏移。"""

    name: str
    type: str
    is_array: bool
    offset: int = 0
    size: int = 0
    value: Any = None
    # 数组专用
    count: int = 0
    count_offset: int = 0
    elements: list["FieldNode"] = field(default_factory=list)


@dataclass
class InstanceNode:
    index: int
    type_hash: int
    name: str
    is_userdata: bool
    start: int = 0
    end: int = 0
    fields: list[FieldNode] = field(default_factory=list)
    userdata_path: str = ""

    def find(self, field_name: str) -> FieldNode | None:
        for f in self.fields:
            if f.name == field_name:
                return f
        return None


@dataclass
class RSZModel:
    rsz_start: int
    data_start: int
    object_count: int
    instances: list[InstanceNode | None]
    userdata_indices: set[int]

    def iter_instances(self):
        for inst in self.instances:
            if inst is not None:
                yield inst

    def root_instances(self) -> list[InstanceNode]:
        """根 instance = 实例表末尾的 object_count 个（RSZ 约定：子节点先于父节点存储）。"""
        total = len(self.instances)
        start_idx = max(1, total - self.object_count)
        roots = []
        for i in range(start_idx, total):
            inst = self.instances[i]
            if inst is not None:
                roots.append(inst)
        return roots


# ---------- 调试 JSON 构建 ----------


def _dbg_field(f: FieldNode, model: RSZModel, visited: set[int]) -> Any:
    """
    将 FieldNode 转为调试 JSON 值。
    - 数组   -> {"__count": N, "__count_off": "0x...", "elements": [...]}
    - Object -> 递归展开为 {ClassName: {...}}（与 dump JSON 一致）
    - 标量   -> {"__v": <值>, "__o": "0x<偏移>"}
    """
    if f.is_array:
        return {
            "__count": f.count,
            "__count_off": hex(f.count_offset),
            "elements": [_dbg_field(e, model, visited) for e in f.elements],
        }

    if f.type in ("Object", "UserData", "Resource"):
        ref_idx = int(f.value) if f.value is not None else 0
        if 0 < ref_idx < len(model.instances):
            ref = model.instances[ref_idx]
            if ref is not None and ref.index not in visited:
                return _dbg_inst(ref, model, visited)
        return {"__ref": ref_idx, "__ref_off": hex(f.offset)}

    v = f.value
    if isinstance(v, bytes):
        v = v.hex()
    return {"__v": v, "__o": hex(f.offset)}


def _dbg_inst(inst: InstanceNode, model: RSZModel, visited: set[int]) -> dict:
    """将 InstanceNode 序列化为 {ClassName: {字段...}} 并附带调试元数据。"""
    if inst.is_userdata:
        return {
            inst.name: {
                "__instance_index": inst.index,
                "__userdata_path": inst.userdata_path,
            }
        }

    visited = visited | {inst.index}
    body: dict[str, Any] = {
        "__instance_index": inst.index,
        "__start": hex(inst.start),
    }
    for f in inst.fields:
        body[f.name] = _dbg_field(f, model, visited)
    return {inst.name: body}


def build_debug_json_path(input_file: Path) -> Path:
    """调试 JSON 保存到 .temp，保持 natives/... 嵌套结构，后缀 .debug.json。"""
    parts = list(input_file.parts)
    if "natives" in parts:
        idx = parts.index("natives")
        rel = Path(*parts[idx:])
        out = Path(".temp") / rel.parent / f"{rel.name}.debug.json"
    else:
        out = Path(".temp") / f"{input_file.name}.debug.json"
    return out


def save_rsz_debug_json(model: RSZModel, input_file: Path) -> Path:
    """
    将解析树序列化为层次化 JSON 写入 .temp。
    结构与 data/json/... 的 dump JSON 对齐（根节点内联所有子节点），
    每个字段值携带 __o（二进制偏移）。
    """
    roots = model.root_instances()
    tree = [_dbg_inst(r, model, set()) for r in roots]
    out_path = build_debug_json_path(input_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(tree, fp, indent=2, ensure_ascii=False)
    return out_path


# ---------- 解析器 ----------


class RSZParser:
    def __init__(self, buf: bytearray, typedb: TypeDB):
        self.buf = buf
        self.typedb = typedb
        self.warn_unknown_classes: set[int] = set()

    def _u8(self, pos: int) -> int:
        return self.buf[pos]

    def _u32(self, pos: int) -> int:
        return struct.unpack_from("<I", self.buf, pos)[0]

    def _i32(self, pos: int) -> int:
        return struct.unpack_from("<i", self.buf, pos)[0]

    def _u64(self, pos: int) -> int:
        return struct.unpack_from("<Q", self.buf, pos)[0]

    def _i64(self, pos: int) -> int:
        return struct.unpack_from("<q", self.buf, pos)[0]

    def _read_wstring(self, pos: int) -> tuple[str, int]:
        """读取 length-prefixed UTF-16 字符串，返回 (字符串, 占用总字节数)。"""
        length = self._u32(pos)
        if length <= 0 or length > 4096:
            return "", 4
        byte_len = length * 2
        raw = bytes(self.buf[pos + 4 : pos + 4 + byte_len])
        try:
            s = raw.decode("utf-16-le", errors="replace").rstrip("\x00")
        except Exception:
            s = ""
        return s, 4 + byte_len

    def _read_guid_str(self, pos: int) -> str:
        """16 字节小端 Guid -> 标准字符串。"""
        return str(uuid.UUID(bytes_le=bytes(self.buf[pos : pos + 16])))

    def parse(self) -> RSZModel:
        usr_magic = self._u32(0)
        if usr_magic not in (0x00525355, 0x52535553):
            raise RuntimeError(f"Not a USR file (magic={usr_magic:#010x})")

        rs = self._u64(32)
        rsz_magic = self._u32(rs)
        if rsz_magic not in (0x005A5352, 0x5A5352):
            raise RuntimeError(f"Not an RSZ block (magic={rsz_magic:#010x})")

        # RSZ header layout (offsets relative to rs):
        #  +0  magic (u32)
        #  +4  version (u32)
        #  +8  object_count (u32)   <- 根对象数，末尾 object_count 个 instance 为根
        # +12  instance_count (u32)
        # +16  userdata_count (u32)
        # +20  reserved (u32)
        # +24  instance_offset (u64)
        # +32  data_offset (u64)
        # +40  userdata_offset (u64)
        object_count = self._u32(rs + 8)
        inst_count = self._i32(rs + 12)
        userdata_count = self._i32(rs + 16)
        inst_off = self._i64(rs + 24)
        data_off = self._i64(rs + 32)
        userdata_off = self._i64(rs + 40)

        # instance info 表：每条 8 字节 = type_hash(u32) + crc(u32)
        inst_info: list[tuple[int, int]] = []
        for i in range(inst_count):
            base = rs + inst_off + i * 8
            inst_info.append((self._u32(base), self._u32(base + 4)))

        # userdata 表：每条 16 字节 = instance_id(u32) + hash(u32) + path_offset(u64)
        userdata_indices: set[int] = set()
        userdata_paths: dict[int, str] = {}
        for i in range(userdata_count):
            rec = rs + userdata_off + i * 16
            uidx = self._i32(rec)
            poff = self._i64(rec + 8)
            path, _ = self._read_wstring(rs + poff) if poff > 0 else ("", 0)
            userdata_indices.add(uidx)
            userdata_paths[uidx] = path

        # 解析数据流：从 data_offset 开始按 instance 顺序填充字段。
        instances: list[InstanceNode | None] = [None] * inst_count
        cursor = rs + data_off
        for idx in range(inst_count):
            type_hash, _crc = inst_info[idx]
            if idx == 0:
                instances[0] = InstanceNode(
                    index=0,
                    type_hash=type_hash,
                    name="<null>",
                    is_userdata=False,
                    start=cursor,
                    end=cursor,
                )
                continue

            cls = self.typedb.get(type_hash)
            cls_name = cls.name if cls else f"<unknown:{type_hash:#010x}>"

            if idx in userdata_indices:
                # UserData 在主数据流中不占任何空间
                instances[idx] = InstanceNode(
                    index=idx,
                    type_hash=type_hash,
                    name=cls_name,
                    is_userdata=True,
                    start=cursor,
                    end=cursor,
                    userdata_path=userdata_paths.get(idx, ""),
                )
                continue

            if cls is None:
                if type_hash not in self.warn_unknown_classes:
                    self.warn_unknown_classes.add(type_hash)
                    print(
                        f"[WARN] Unknown class hash {type_hash:#010x} at instance {idx} "
                        f"(cursor={cursor:#x}); skipping."
                    )
                instances[idx] = InstanceNode(
                    index=idx,
                    type_hash=type_hash,
                    name=cls_name,
                    is_userdata=False,
                    start=cursor,
                    end=cursor,
                )
                continue

            if not cls.fields:
                instances[idx] = InstanceNode(
                    index=idx,
                    type_hash=type_hash,
                    name=cls_name,
                    is_userdata=False,
                    start=cursor,
                    end=cursor,
                )
                continue

            first = cls.fields[0]
            cursor = _align(cursor, 4 if first.is_array else first.align)
            start = cursor
            field_nodes, cursor = self._parse_fields(cursor, cls)
            instances[idx] = InstanceNode(
                index=idx,
                type_hash=type_hash,
                name=cls_name,
                is_userdata=False,
                start=start,
                end=cursor,
                fields=field_nodes,
            )

        return RSZModel(
            rsz_start=rs,
            data_start=rs + data_off,
            object_count=object_count,
            instances=instances,
            userdata_indices=userdata_indices,
        )

    def _parse_fields(self, cursor: int, cls: ClassDef) -> tuple[list[FieldNode], int]:
        nodes: list[FieldNode] = []
        for fld in cls.fields:
            node, cursor = self._parse_one_field(cursor, fld, cls)
            nodes.append(node)
        return nodes, cursor

    def _parse_one_field(
        self, cursor: int, fld: FieldDef, owner_cls: ClassDef
    ) -> tuple[FieldNode, int]:
        if fld.is_array:
            cursor = _align(cursor, 4)
            count = self._i32(cursor)
            count_offset = cursor
            cursor += 4

            if count < 0 or count > 1_000_000:
                print(
                    f"[WARN] Suspicious array count={count} at {count_offset:#x} "
                    f"(class={owner_cls.name} field={fld.name}); skipping array body."
                )
                return (
                    FieldNode(
                        name=fld.name,
                        type=fld.type,
                        is_array=True,
                        count=0,
                        count_offset=count_offset,
                        offset=count_offset,
                    ),
                    cursor,
                )

            elements: list[FieldNode] = []
            data_offset = cursor
            if count > 0:
                cursor = _align(cursor, fld.align)
                data_offset = cursor
                for _ in range(count):
                    elem_node, cursor = self._read_value(cursor, fld)
                    elements.append(elem_node)
            return (
                FieldNode(
                    name=fld.name,
                    type=fld.type,
                    is_array=True,
                    count=count,
                    count_offset=count_offset,
                    offset=data_offset,
                    elements=elements,
                ),
                cursor,
            )
        else:
            cursor = _align(cursor, fld.align)
            node, cursor = self._read_value(cursor, fld)
            node.name = fld.name
            node.is_array = False
            return node, cursor

    def _read_value(self, cursor: int, fld: FieldDef) -> tuple[FieldNode, int]:
        """读取一个非数组的值（cursor 已对齐）。"""
        ftype = fld.type
        offset = cursor

        if ftype in _VARLEN_STRING_TYPES:
            text, used = self._read_wstring(cursor)
            return (
                FieldNode(
                    name=fld.name,
                    type=ftype,
                    is_array=False,
                    offset=offset,
                    size=used,
                    value=text,
                ),
                cursor + used,
            )

        size = fld.size if fld.size > 0 else _SCALAR_SIZES.get(ftype, 4)

        if ftype in ("Guid", "GameObjectRef", "Uri"):
            value: Any = self._read_guid_str(cursor)
        elif ftype == "Bool":
            value = bool(self.buf[cursor])
        elif ftype == "S8":
            value = struct.unpack_from("<b", self.buf, cursor)[0]
        elif ftype == "U8":
            value = self.buf[cursor]
        elif ftype == "S16":
            value = struct.unpack_from("<h", self.buf, cursor)[0]
        elif ftype == "U16":
            value = struct.unpack_from("<H", self.buf, cursor)[0]
        elif ftype in ("S32", "Sfix", "Enum"):
            value = struct.unpack_from("<i", self.buf, cursor)[0]
        elif ftype == "U32":
            value = struct.unpack_from("<I", self.buf, cursor)[0]
        elif ftype == "S64":
            value = struct.unpack_from("<q", self.buf, cursor)[0]
        elif ftype == "U64":
            value = struct.unpack_from("<Q", self.buf, cursor)[0]
        elif ftype == "F32":
            value = struct.unpack_from("<f", self.buf, cursor)[0]
        elif ftype == "F64":
            value = struct.unpack_from("<d", self.buf, cursor)[0]
        elif ftype in ("Object", "UserData", "Resource"):
            value = struct.unpack_from("<I", self.buf, cursor)[0]
        else:
            value = bytes(self.buf[cursor : cursor + size])

        return (
            FieldNode(
                name=fld.name,
                type=ftype,
                is_array=False,
                offset=offset,
                size=size,
                value=value,
            ),
            cursor + size,
        )


# ---------- 修改逻辑 ----------

_PATCH_LIMIT_CLASSES = (
    "app.MissionUserdata.Unit.ConditionUnit",
    "app.MissionUserdata.Unit.ConditionUnitBase",
)

# Unit.Conditions 里若存在 Hash 为该值的 ConditionUnit，则认定此 Unit 适用 no-damage 规则
# （对该 Unit 下 Conditions 中每条 ConditionUnit 写 LimitValue=0、LimitValueF=limit）。
NO_DAMAGE_UNIT_MARKER_HASH = 2168305544
DESC_OVERRIDE_HASH = 2168305544
DESC_OVERRIDE_MSGID = "5b903120-f93a-4b37-8d0f-e1b12d54e03c"
DESC2_OVERRIDE_MSGID = "47e19cda-2415-4c48-91a0-cc3086912eea"


def _success_conditions_base_indices(model: RSZModel) -> set[int]:
    """每个 Unit 的 SuccessConditions 数组里引用的 instance 下标（多为 ConditionUnitBase）。"""
    out: set[int] = set()
    for inst in model.iter_instances():
        if inst.name != "app.MissionUserdata.Unit":
            continue
        f = inst.find("SuccessConditions")
        if f is None or not f.is_array:
            continue
        for elem in f.elements:
            out.add(int(elem.value))
    return out


def _success_paired_condition_unit_indices(
    model: RSZModel, target_hashes: set[int]
) -> set[int]:
    """
    若某 Unit 的 SuccessConditions 里存在 Hash∈target_hashes 的 ConditionUnitBase，
    则在该 Unit 的 Conditions 数组中，按顺序找第一个 Hash∈target_hashes 的 ConditionUnit，
    其 instance 下标加入保护集（与通关 Success 成对展示的那条，LimitValueF 也不改）。
    """
    protected: set[int] = set()
    for inst in model.iter_instances():
        if inst.name != "app.MissionUserdata.Unit":
            continue
        suc = inst.find("SuccessConditions")
        if suc is None or not suc.is_array:
            continue
        has_success_target = False
        for elem in suc.elements:
            ref = int(elem.value)
            if not (0 <= ref < len(model.instances)):
                continue
            base_inst = model.instances[ref]
            if base_inst is None:
                continue
            if base_inst.name != "app.MissionUserdata.Unit.ConditionUnitBase":
                continue
            hf = base_inst.find("Hash")
            if hf is None:
                continue
            if int(hf.value) & 0xFFFFFFFF in target_hashes:
                has_success_target = True
                break
        if not has_success_target:
            continue
        cond = inst.find("Conditions")
        if cond is None or not cond.is_array:
            continue
        for elem in cond.elements:
            ref = int(elem.value)
            if not (0 <= ref < len(model.instances)):
                continue
            cu = model.instances[ref]
            if cu is None or cu.name != "app.MissionUserdata.Unit.ConditionUnit":
                continue
            hf = cu.find("Hash")
            if hf is None:
                continue
            if int(hf.value) & 0xFFFFFFFF in target_hashes:
                protected.add(ref)
                break
    return protected


def patch_condition_units(
    buf: bytearray,
    model: RSZModel,
    target_hashes: set[int],
    patch_value: float,
    *,
    skip_success_condition_bases: bool = True,
) -> int:
    """
    把 Hash 在 target_hashes 中的 LimitValueF 写入 patch_value，适用于：
      - app.MissionUserdata.Unit.ConditionUnit（任务 Conditions 列表）
      - app.MissionUserdata.Unit.ConditionUnitBase（SuccessConditions / FailConditions 等）

    默认 skip_success_condition_bases=True：不修改任何 Unit.SuccessConditions 引用的
    ConditionUnitBase（避免动到「通关判定」阈值）；FailConditions 等处的 Base 仍会修改。
    同时：若某 Unit 的 Success 里已有目标 Hash，则该 Unit.Conditions 中「首个」目标 Hash
    的 ConditionUnit 也跳过（与 Success 成对的那条 UI/逻辑）。
    与 dump JSON 统计一致时：57×ConditionUnit + 17×Base ≈ 74；加 --patch-success-base 可凑满 76。
    """
    protected_base: set[int] = set()
    protected_cu_success_pair: set[int] = set()
    if skip_success_condition_bases:
        protected_base = _success_conditions_base_indices(model)
        protected_cu_success_pair = _success_paired_condition_unit_indices(
            model, target_hashes
        )
        if protected_base:
            print(
                f"[INFO] Skip {len(protected_base)} instance(s) "
                f"referenced by SuccessConditions (ConditionUnitBase only)."
            )
        if protected_cu_success_pair:
            print(
                f"[INFO] Skip {len(protected_cu_success_pair)} ConditionUnit(s) "
                f"(first target-hash entry per Unit when Success uses target hash)."
            )

    patched = 0
    for inst in model.iter_instances():
        if inst.name not in _PATCH_LIMIT_CLASSES:
            continue

        if inst.name == "app.MissionUserdata.Unit.ConditionUnitBase":
            if inst.index in protected_base:
                continue

        hash_f = inst.find("Hash")
        limit_f = inst.find("LimitValueF")
        if hash_f is None or limit_f is None:
            continue

        h = int(hash_f.value) & 0xFFFFFFFF
        if h not in target_hashes:
            continue

        if (
            inst.name == "app.MissionUserdata.Unit.ConditionUnit"
            and inst.index in protected_cu_success_pair
        ):
            before = struct.unpack_from("<f", buf, limit_f.offset)[0]
            print(
                f"  [SKIP] [CU] inst[{inst.index}] hash={h} @ {inst.start:#x} "
                f"paired with SuccessConditions target hash; LimitValueF kept at {before}"
            )
            continue

        tag = "CU" if inst.name.endswith(".ConditionUnit") else "CUB"
        before = struct.unpack_from("<f", buf, limit_f.offset)[0]
        struct.pack_into("<f", buf, limit_f.offset, float(patch_value))
        print(
            f"  [+] [{tag}] inst[{inst.index}] hash={h} @ {inst.start:#x} "
            f"LimitValueF {before} -> {patch_value} (offset={limit_f.offset:#x})"
        )
        patched += 1

    return patched


def patch_no_damage_conditions(
    buf: bytearray,
    model: RSZModel,
    limit_value: float,
    *,
    marker_hash: int = NO_DAMAGE_UNIT_MARKER_HASH,
) -> int:
    """
    对所有 MissionUserdata.Unit 的 Conditions：
    仅当某条 ConditionUnit 的 Hash == marker_hash（默认 2168305544）时，
    将该条 ConditionUnit 的 LimitValue=0、LimitValueF=limit_value。

    返回值单独统计，不计入 target-hash 补丁总数。
    """
    n = 0
    mh = int(marker_hash) & 0xFFFFFFFF
    for inst in model.iter_instances():
        if inst.name != "app.MissionUserdata.Unit":
            continue
        cond_field = inst.find("Conditions")
        if cond_field is None or not cond_field.is_array:
            continue
        for elem in cond_field.elements:
            ref = int(elem.value)
            if not (0 <= ref < len(model.instances)):
                continue
            cu = model.instances[ref]
            if cu is None or cu.name != "app.MissionUserdata.Unit.ConditionUnit":
                continue
            hf = cu.find("Hash")
            if hf is None or (int(hf.value) & 0xFFFFFFFF) != mh:
                continue
            lv = cu.find("LimitValue")
            lvf = cu.find("LimitValueF")
            if lv is None or lvf is None:
                continue
            before_i = struct.unpack_from("<i", buf, lv.offset)[0]
            before_f = struct.unpack_from("<f", buf, lvf.offset)[0]
            struct.pack_into("<i", buf, lv.offset, 0)
            struct.pack_into("<f", buf, lvf.offset, float(limit_value))
            print(
                f"  [NO-DAMAGE] Unit inst[{inst.index}] -> CU inst[{ref}] "
                f"LimitValue {before_i} -> 0 (off={lv.offset:#x}), "
                f"LimitValueF {before_f} -> {limit_value} (off={lvf.offset:#x})"
            )
            n += 1
    return n


def _guid_str_to_le_bytes(guid_str: str) -> bytes:
    try:
        return uuid.UUID(guid_str).bytes_le
    except Exception:
        return b""


def patch_desc_msgids_for_hash_2168305544(
    buf: bytearray,
    model: RSZModel,
) -> int:
    """
    对 Hash==2168305544 的 ConditionUnit：
      DescText._MessageId    -> 5b903120-f93a-4b37-8d0f-e1b12d54e03c
      DescText2Line._MessageId -> 47e19cda-2415-4c48-91a0-cc3086912eea
    返回修改了多少个 ConditionUnit（按条计，不按字段计）。
    """
    msg1 = _guid_str_to_le_bytes(DESC_OVERRIDE_MSGID)
    msg2 = _guid_str_to_le_bytes(DESC2_OVERRIDE_MSGID)
    if len(msg1) != 16 or len(msg2) != 16:
        print("[WARN] DescText override GUID invalid, skip desc override.")
        return 0

    patched_units = 0
    for inst in model.iter_instances():
        if inst.name != "app.MissionUserdata.Unit.ConditionUnit":
            continue
        hash_f = inst.find("Hash")
        if hash_f is None:
            continue
        if (int(hash_f.value) & 0xFFFFFFFF) != DESC_OVERRIDE_HASH:
            continue

        changed_any = False
        for field_name, guid_bytes in (("DescText", msg1), ("DescText2Line", msg2)):
            f = inst.find(field_name)
            if f is None:
                continue
            ref_idx = int(f.value)
            if not (0 <= ref_idx < len(model.instances)):
                continue
            text_inst = model.instances[ref_idx]
            if text_inst is None or text_inst.name != "app.TextMessageData":
                continue
            msg_field = text_inst.find("_MessageId")
            if msg_field is None:
                continue
            before = bytes(buf[msg_field.offset : msg_field.offset + 16])
            if before != guid_bytes:
                buf[msg_field.offset : msg_field.offset + 16] = guid_bytes
                changed_any = True
                print(
                    f"  [DESC] CU inst[{inst.index}] {field_name}._MessageId "
                    f"@ {msg_field.offset:#x} -> "
                    f"{(DESC_OVERRIDE_MSGID if field_name == 'DescText' else DESC2_OVERRIDE_MSGID)}"
                )

        if changed_any:
            patched_units += 1

    return patched_units


# ---------- I/O ----------


def build_default_output_path(input_file: Path) -> Path:
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
    target_hashes: int | list[int] = 3410781912,
    patch_value: float = 99999.0,
    dump_debug: bool = True,
    patch_success_condition_bases: bool = False,
    no_damage: bool = False,
) -> bool:
    if not input_file.exists():
        print(f"[ERROR] File not found: {input_file}")
        return False

    print(f"Loading TypeDB from {rsz_path}...")
    typedb = TypeDB(rsz_path)
    if not typedb.classes:
        return False

    if isinstance(target_hashes, int):
        target_hash_set = {int(target_hashes) & 0xFFFFFFFF}
    else:
        target_hash_set = {int(x) & 0xFFFFFFFF for x in target_hashes}

    print(f"Reading {input_file}...")
    try:
        buf = bytearray(input_file.read_bytes())
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return False

    print("Parsing RSZ structure (deriving layout with offsets)...")
    parser = RSZParser(buf, typedb)
    try:
        model = parser.parse()
    except Exception as e:
        print(f"[ERROR] RSZ parse failed: {e}")
        return False

    if dump_debug:
        try:
            debug_path = save_rsz_debug_json(model, input_file)
            print(f"[DEBUG] Layout JSON -> {debug_path}")
        except Exception as e:
            print(f"[WARN] Failed to write debug JSON: {e}")

    n_cu = sum(
        1
        for inst in model.iter_instances()
        if inst.name == "app.MissionUserdata.Unit.ConditionUnit"
    )
    n_cub = sum(
        1
        for inst in model.iter_instances()
        if inst.name == "app.MissionUserdata.Unit.ConditionUnitBase"
    )
    print(
        f"[INFO] Parsed {sum(1 for _ in model.iter_instances())} instances, "
        f"{n_cu} ConditionUnit + {n_cub} ConditionUnitBase; "
        f"userdata refs={len(model.userdata_indices)}"
    )

    patched = patch_condition_units(
        buf,
        model,
        target_hash_set,
        patch_value,
        skip_success_condition_bases=not patch_success_condition_bases,
    )
    print(f"\nTotal patched (target hash, LimitValueF): {patched}")

    desc_override_n = 0
    if no_damage:
        print("\n--- DescText override (Hash=2168305544, tied to no-damage) ---")
        desc_override_n = patch_desc_msgids_for_hash_2168305544(buf, model)
        print(f"Desc override patched (separate count): {desc_override_n}")
    else:
        print("\n--- DescText override ---")
        print("Desc override skipped (enable with --no-damage).")

    no_damage_n = 0
    if no_damage:
        print("\n--- No-damage (Conditions: LimitValue=0 + LimitValueF=limit) ---")
        no_damage_n = patch_no_damage_conditions(buf, model, patch_value)
        print(f"No-damage patched (separate count): {no_damage_n}")

    if patched <= 0 and no_damage_n <= 0 and desc_override_n <= 0:
        print("Nothing patched, output file not written.")
        return False

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(buf)
        print(f"Saved modified file to {output_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write file: {e}")
        return False


# ---------- 入口 ----------


def main():
    p = argparse.ArgumentParser(
        description="Patch missionuserdata.user.3 LimitValueF based on Hash."
    )
    p.add_argument("-i", "--input", type=str, required=True)
    p.add_argument("-o", "--output", type=str, default=None)
    p.add_argument("--rsz", type=str, default="data/rszpragmata.json")
    p.add_argument(
        "--hash",
        type=int,
        nargs="+",
        default=[3410781912, 3115479535],
        help="Target Hash list (space-separated).",
    )
    p.add_argument("--limit", type=float, default=99999.0)
    p.add_argument(
        "--no-dump-debug",
        action="store_true",
        help="Do not write .temp/.../*.debug.json.",
    )
    p.add_argument(
        "--patch-success-base",
        action="store_true",
        help="Also patch ConditionUnitBase under SuccessConditions (default: skip).",
    )
    p.add_argument(
        "--no-damage",
        action="store_true",
        help=(
            "For ConditionUnit entries whose Hash is "
            f"{NO_DAMAGE_UNIT_MARKER_HASH}, set LimitValue=0 and "
            "LimitValueF to --limit. Logged separately."
        ),
    )

    args = p.parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output) if args.output else build_default_output_path(input_path)
    )
    patch_mission_user_data(
        input_path,
        output_path,
        Path(args.rsz),
        args.hash,
        args.limit,
        dump_debug=not args.no_dump_debug,
        patch_success_condition_bases=args.patch_success_base,
        no_damage=args.no_damage,
    )


if __name__ == "__main__":
    main()
