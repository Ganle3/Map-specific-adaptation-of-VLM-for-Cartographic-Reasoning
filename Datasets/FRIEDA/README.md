---
task_categories:
- visual-question-answering
language:
- en
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: data
    path: data/data-*
dataset_info:
  features:
  - name: question_ref
    dtype: string
  - name: images
    list: string
  - name: question_text
    dtype: string
  - name: expected_answer
    dtype: string
  - name: map_count
    dtype: string
  - name: spatial_relationship
    dtype: string
  - name: answer_type
    dtype: string
  - name: domain
    dtype: string
  - name: map_elements
    list: string
  - name: context_images
    list: string
  splits:
  - name: data
    num_bytes: 576010
    num_examples: 500
  download_size: 120293
  dataset_size: 576010
pretty_name: FRIEDA
---

[![arXiv](https://img.shields.io/badge/arXiv-2512.08016-111111?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.08016)
[![Website](https://img.shields.io/badge/Website-Webpage-111111?style=for-the-badge&logo=googlechrome&logoColor=white)](https://knowledge-computing.github.io/FRIEDA/)
[![Code](https://img.shields.io/badge/Code-GitHub-111111?style=for-the-badge&logo=github&logoColor=white)](https://github.com/knowledge-computing/FRIEDA)

**FRIEDA** is a multimodal benchmark for **open-ended cartographic reasoning** over real-world map images.  
Each example pairs reference maps (and optional contextual maps) with a natural-language question and a reference answer. The benchmark targets common GIS relation types (i.e., **topological**, **metric**, **directional**) and includes questions that require multi-step reasoning and cross-map grounding.

### Dataset Summary

- **Modality:** image + text  
- **# Examples:** 500  
- **Input:** map image(s) + question text  
- **Output:** expected answer (textual)
- **Metadata:** map_count, domain, relationship type, map elements

### Languages

The dataset questions and answers are in **English**.

---

## How to use it

```python
from datasets import load_dataset

# Full dataset (split name = "data")
ds = load_dataset("knowledge-computing/FRIEDA", split="data")
print(ds[0].keys())
print(ds[0]["question_text"])    # Actual question being asked
print(ds[0]["images"])           # List of string paths to images (e.g., "images/...png")
print(ds[0]["context_images"])   # List of string paths to contextual images