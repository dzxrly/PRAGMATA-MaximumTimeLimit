import sys
from pathlib import Path

# 确保能正确导入当前目录的 main.py
sys.path.append(str(Path(__file__).parent))

from main import patch_mission_user_data


def debug():
    """
    开发调试专用入口。
    直接使用项目默认路径读取文件，并将结果输出到 .temp 临时目录中，免去命令行输入长串参数的麻烦。
    """
    # 默认输入文件路径（原版文件）
    input_file = Path("data/user3/natives/stm/singletonuserdata/missionuserdata.user.3")

    # 调试输出目录，使用 .temp 避免污染正式构建的 data/output 目录
    output_file = Path(".temp/natives/stm/singletonuserdata/missionuserdata.user.3")

    # RSZ 模板路径
    rsz_path = Path("data/rszpragmata.json")

    print("=" * 50)
    print("🚀 [DEBUG MODE] EasyMiniGame Patch Runner")
    print("=" * 50)
    print(f"[*] Input : {input_file}")
    print(f"[*] Output: {output_file}")
    print(f"[*] RSZ   : {rsz_path}")
    print("-" * 50)

    success = patch_mission_user_data(
        input_file, output_file, rsz_path, [3410781912, 3115479535], 99999.0
    )

    print("-" * 50)
    if success:
        print(f"✅ Debug patch successful!")
        print(f"📂 Check the patched file at: {output_file}")
    else:
        print("❌ Debug patch failed or no modifications were made.")
    print("=" * 50)


if __name__ == "__main__":
    debug()
