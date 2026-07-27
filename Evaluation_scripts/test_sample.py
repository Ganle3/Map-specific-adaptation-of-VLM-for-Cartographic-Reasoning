from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Unsloth 必须先于 transformers 导入。
# 即使这里使用标准 Transformers 加载模型，也让 Unsloth 先完成优化 patch。
import unsloth

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


# ============================================================
# 1. 用户配置
# ============================================================

# 可选："frieda" 或 "mapwise"
DATASET = "frieda"

# 读取 JSON 中第几条数据，0 表示第一条。
SAMPLE_INDEX = 0

# MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"
MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"

# Thinking 模型需要足够的生成长度。
MAX_NEW_TOKENS = 1536


# ============================================================
# 2. 数据路径
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_DATA_ROOT = PROJECT_ROOT / "Test_data"

FRIEDA_ROOT = TEST_DATA_ROOT / "FRIEDA_test"
FRIEDA_JSON = FRIEDA_ROOT / "frieda_test.json"
FRIEDA_IMAGE_ROOT = FRIEDA_ROOT / "image"

MAPWISE_ROOT = TEST_DATA_ROOT / "Mapwise_india"
MAPWISE_JSON = MAPWISE_ROOT / "india_test_75_balanced.json"
MAPWISE_IMAGE_ROOT = MAPWISE_ROOT / "image"


# MapWise 图片可能使用这些后缀。
IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


# ============================================================
# 3. 通用工具函数
# ============================================================

def load_json_list(json_path: Path) -> list[dict[str, Any]]:
    """读取顶层为 list 的 JSON 文件。"""
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON 文件不存在：\n{json_path}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            f"JSON 顶层应当是 list，但实际是：{type(data).__name__}"
        )

    if not data:
        raise ValueError(f"JSON 文件为空：{json_path}")

    return data


def validate_image(image_path: Path) -> None:
    """检查图片文件是否存在且 PIL 能正常读取。"""
    if not image_path.exists():
        raise FileNotFoundError(
            f"图片文件不存在：\n{image_path}"
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"图片路径不是文件：\n{image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        raise RuntimeError(
            f"图片无法正常打开：\n{image_path}\n"
            f"原始错误：{error}"
        ) from error


def print_image_information(image_paths: list[Path]) -> None:
    """打印图片尺寸和格式。"""
    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            print(
                f"Image {index}: {image_path}\n"
                f"  format = {image.format}\n"
                f"  size   = {image.width} x {image.height}\n"
                f"  mode   = {image.mode}"
            )


# ============================================================
# 4. FRIEDA 路径与样本适配
# ============================================================

def resolve_frieda_images(
    sample: dict[str, Any],
    image_root: Path,
) -> list[Path]:
    """
    FRIEDA 中 image_urls 的路径相对于：
    Test_data/FRIEDA_test/image/

    例如：
    image_urls =
    [
        "NI43-101/.../image65_2.jpeg"
    ]

    最终路径为：
    Test_data/FRIEDA_test/image/NI43-101/.../image65_2.jpeg
    """
    image_urls = sample.get("image_urls")

    if not isinstance(image_urls, list) or not image_urls:
        raise ValueError(
            "FRIEDA 样本缺少有效的 image_urls 列表。"
        )

    image_paths: list[Path] = []

    for relative_path in image_urls:
        normalized_relative_path = str(relative_path).replace("\\", "/")
        image_path = image_root / Path(normalized_relative_path)

        validate_image(image_path)
        image_paths.append(image_path.resolve())

    return image_paths


def load_frieda_sample(
    sample_index: int,
) -> tuple[dict[str, Any], list[Path], str, str, str]:
    data = load_json_list(FRIEDA_JSON)

    if not 0 <= sample_index < len(data):
        raise IndexError(
            f"FRIEDA SAMPLE_INDEX={sample_index} 超出范围。"
            f"有效范围是 0 到 {len(data) - 1}。"
        )

    sample = data[sample_index]

    sample_id = str(
        sample.get("question_ref", f"frieda_{sample_index}")
    )
    question = str(sample["question_text"]).strip()
    expected_answer = str(sample["expected_answer"]).strip()

    image_paths = resolve_frieda_images(
        sample=sample,
        image_root=FRIEDA_IMAGE_ROOT,
    )

    return (
        sample,
        image_paths,
        sample_id,
        question,
        expected_answer,
    )


# ============================================================
# 5. MapWise 路径与样本适配
# ============================================================

def find_mapwise_image(
    map_no: str,
    image_root: Path,
) -> Path:
    """
    根据 map_no 查找图片。

    例如：
        map_no = "map101_2D"

    尝试：
        image/map101_2D.png
        image/map101_2D.jpg
        image/map101_2D.jpeg
        ...

    如果图片位于 image 的更深层目录，也会递归搜索。
    """
    map_no_path = Path(map_no)

    # 情况 1：map_no 自己已经包含扩展名。
    if map_no_path.suffix:
        direct_path = image_root / map_no_path

        if direct_path.exists():
            validate_image(direct_path)
            return direct_path.resolve()

    # 情况 2：map_no 不包含扩展名，在 image 根目录尝试常见后缀。
    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / f"{map_no}{extension}"

        if candidate.exists():
            validate_image(candidate)
            return candidate.resolve()

    # 情况 3：图片可能放在 image 下的子文件夹中，递归搜索。
    recursive_matches: list[Path] = []

    for extension in IMAGE_EXTENSIONS:
        recursive_matches.extend(
            image_root.rglob(f"{map_no}{extension}")
        )

    # 去重。
    recursive_matches = sorted(
        set(path.resolve() for path in recursive_matches)
    )

    if len(recursive_matches) == 1:
        validate_image(recursive_matches[0])
        return recursive_matches[0]

    if len(recursive_matches) > 1:
        match_text = "\n".join(
            str(path) for path in recursive_matches
        )
        raise RuntimeError(
            f"找到多个与 map_no={map_no!r} 对应的图片：\n"
            f"{match_text}\n"
            "请检查 MapWise image 文件夹是否存在重复文件。"
        )

    available_examples = [
        path.name
        for path in image_root.iterdir()
        if path.is_file()
    ][:10]

    raise FileNotFoundError(
        f"找不到 map_no={map_no!r} 对应的图片。\n"
        f"搜索目录：{image_root}\n"
        f"已尝试后缀：{IMAGE_EXTENSIONS}\n"
        f"目录中的前几个文件：{available_examples}"
    )


def load_mapwise_sample(
    sample_index: int,
) -> tuple[dict[str, Any], list[Path], str, str, str]:
    data = load_json_list(MAPWISE_JSON)

    if not 0 <= sample_index < len(data):
        raise IndexError(
            f"MapWise SAMPLE_INDEX={sample_index} 超出范围。"
            f"有效范围是 0 到 {len(data) - 1}。"
        )

    sample = data[sample_index]

    sample_id = str(
        sample.get("qa_id", f"mapwise_{sample_index}")
    )
    question = str(sample["question"]).strip()
    expected_answer = str(sample["ground_truth"]).strip()
    map_no = str(sample["map_no"]).strip()

    image_path = find_mapwise_image(
        map_no=map_no,
        image_root=MAPWISE_IMAGE_ROOT,
    )

    return (
        sample,
        [image_path],
        sample_id,
        question,
        expected_answer,
    )


# ============================================================
# 6. 构建模型输入
# ============================================================

def build_prompt(
    dataset_name: str,
    question: str,
) -> str:
    """
    FRIEDA 和 MapWise 都是开放式 QA，不是选择题。
    因此不要要求输出 A/B/C/D。
    """
    if dataset_name == "frieda":
        dataset_instruction = """
This is a cartographic reasoning question from the FRIEDA dataset.
Use only the supplied map image or images.
Carefully inspect map labels, legends, symbols, boundaries,
directions, distances, and spatial relationships.
""".strip()

    elif dataset_name == "mapwise":
        dataset_instruction = """
This is a thematic-map reasoning question from the MapWise dataset.
Use the supplied map to inspect state locations, map annotations,
legend classes, numerical values, colors, and spatial relationships.
""".strip()

    else:
        raise ValueError(f"不支持的数据集：{dataset_name}")

    return f"""
{dataset_instruction}

Question:
{question}

Reason through the question carefully.

At the end, provide the answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()


def build_messages(
    image_paths: list[Path],
    prompt: str,
) -> list[dict[str, Any]]:
    """
    Qwen3-VL 的 content 顺序：
    image 1, image 2, ..., text question
    """
    content: list[dict[str, Any]] = []

    for image_path in image_paths:
        content.append(
            {
                "type": "image",
                "image": str(image_path),
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# ============================================================
# 7. 加载 Qwen3-VL-8B-Thinking
# ============================================================

def load_model_and_processor():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 没有检测到 CUDA GPU。\n"
            f"torch version = {torch.__version__}\n"
            f"torch CUDA build = {torch.version.cuda}"
        )

    print("\nLoading model...", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "GPU VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
        "GB",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.eval()

    print("Model loaded successfully.", flush=True)

    return model, processor


# ============================================================
# 8. 单条推理
# ============================================================

def run_inference(
    model,
    processor,
    messages: list[dict[str, Any]],
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # device_map="auto" 时，输入放到模型首个设备。
    model_device = next(model.parameters()).device

    inputs = {
        key: (
            value.to(model_device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    generated_tokens = generated_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    dataset_name = DATASET.strip().lower()

    if dataset_name == "frieda":
        (
            sample,
            image_paths,
            sample_id,
            question,
            expected_answer,
        ) = load_frieda_sample(SAMPLE_INDEX)

    elif dataset_name == "mapwise":
        (
            sample,
            image_paths,
            sample_id,
            question,
            expected_answer,
        ) = load_mapwise_sample(SAMPLE_INDEX)

    else:
        raise ValueError(
            f'DATASET 必须是 "frieda" 或 "mapwise"，'
            f"当前值为：{DATASET!r}"
        )

    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Sample index: {SAMPLE_INDEX}")
    print(f"Sample ID: {sample_id}")
    print(f"Number of images: {len(image_paths)}")
    print(f"\nQuestion:\n{question}")
    print(f"\nExpected answer:\n{expected_answer}")
    print("\nResolved image paths:")

    for path in image_paths:
        print(f"  {path}")

    print()
    print_image_information(image_paths)

    prompt = build_prompt(
        dataset_name=dataset_name,
        question=question,
    )

    messages = build_messages(
        image_paths=image_paths,
        prompt=prompt,
    )

    model, processor = load_model_and_processor()

    print("\nRunning inference...")

    output_text = run_inference(
        model=model,
        processor=processor,
        messages=messages,
    )

    print("\n" + "=" * 80)
    print("MODEL OUTPUT")
    print("=" * 80)
    print(output_text)

    print("\n" + "=" * 80)
    print("GROUND TRUTH")
    print("=" * 80)
    print(expected_answer)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n程序运行失败：", file=sys.stderr)
        print(
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise