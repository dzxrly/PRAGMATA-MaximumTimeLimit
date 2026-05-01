import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import build_debug_json_path, patch_mission_user_data


def debug():
    """
    开发调试专用入口。
    直接使用项目默认路径读取文件，将结果输出到 .temp 临时目录，免去命令行长串参数。
    同时生成 .debug.json（层次化，与 dump JSON 结构一致，每个值附带二进制偏移）。
    """
    input_file = Path("data/user3/natives/stm/singletonuserdata/missionuserdata.user.3")
    output_file = Path(".temp/natives/stm/singletonuserdata/missionuserdata.user.3")
    rsz_path = Path("data/rszpragmata.json")
    debug_json = build_debug_json_path(input_file)

    print("=" * 50)
    print("[DEBUG MODE] EasyMiniGame Patch Runner")
    print("=" * 50)
    print(f"[*] Input      : {input_file}")
    print(f"[*] Output     : {output_file}")
    print(f"[*] RSZ        : {rsz_path}")
    print(f"[*] Debug JSON : {debug_json}")
    print(f"[*] No-damage  : True (debug default)")
    print("-" * 50)

    success = patch_mission_user_data(
        input_file,
        output_file,
        rsz_path,
        [3410781912, 3115479535],
        99999.0,
        dump_debug=True,
        no_damage=False,
    )

    print("-" * 50)
    if success:
        print(f"[OK] Debug patch successful!")
        print(f"     Patched file : {output_file}")
    else:
        print("[FAIL] Debug patch failed or no modifications were made.")
    if debug_json.is_file():
        print(f"     Debug JSON   : {debug_json}")
    print("=" * 50)


if __name__ == "__main__":
    debug()
