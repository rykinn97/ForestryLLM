import pandas as pd
import json
from pathlib import Path

# =========================
# 1. 输入输出路径
# =========================
excel_path = Path("/root/autodl-tmp/ForestryLLM/Forestry_KB/chunking_datas/B002_M_园林植物遗传育种_切块工作版.xlsx")   # 改成你的 Excel 路径
output_dir = Path("/root/autodl-tmp/ForestryLLM/Forestry_KB/exports_datas/B002_M_园林植物遗传育种")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 中文列名 -> 英文字段名映射
# =========================
column_mapping = {
    "知识块编号": "chunk_id",
    "书名": "book_title",
    "章标题": "chapter_title",
    "节标题": "section_title",
    "起始页码": "page_start",
    "结束页码": "page_end",
    "知识块标题": "chunk_title",
    "内容类型": "topic_type",
    "清洗后正文": "cleaned_text",
    "关键词": "keywords",
    "引用锚点": "citation_anchor"
}

required_columns = [
    "chunk_id",
    "book_title",
    "chapter_title",
    "section_title",
    "page_start",
    "page_end",
    "chunk_title",
    "topic_type",
    "cleaned_text",
    "keywords",
    "citation_anchor"
]

# =========================
# 3. 读取 Excel
# =========================
df = pd.read_excel(excel_path)

# 如果是中文列名，改成英文字段名
df = df.rename(columns=column_mapping)

# 检查缺失列
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Excel 缺少必要列: {missing}")

# 只保留标准列
df = df[required_columns].copy()

# 去掉完全空白行
df = df.dropna(how="all")

# 去掉正文为空的行
df = df[df["cleaned_text"].notna()]

# =========================
# 4. 数据清洗与格式统一
# =========================
for col in ["chunk_id", "book_title", "chapter_title", "section_title",
            "chunk_title", "topic_type", "cleaned_text", "keywords", "citation_anchor"]:
    df[col] = df[col].astype(str).str.strip()

df["topic_type"] = df["topic_type"].str.lower()

df["page_start"] = pd.to_numeric(df["page_start"], errors="coerce").astype("Int64")
df["page_end"] = pd.to_numeric(df["page_end"], errors="coerce").astype("Int64")

# 去重：按 chunk_id 去重，保留第一条
df = df.drop_duplicates(subset=["chunk_id"], keep="first")

# =========================
# 5. 导出文件名
# =========================
base_name = excel_path.stem.replace("_切块工作版", "").replace(" ", "_")
csv_path = output_dir / f"{base_name}_chunks_clean.csv"
jsonl_path = output_dir / f"{base_name}_chunks_clean.jsonl"

# =========================
# 6. 导出 CSV
# =========================
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# =========================
# 7. 导出 JSONL
# =========================
with open(jsonl_path, "w", encoding="utf-8") as f:
    for record in df.to_dict(orient="records"):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("导出完成：")
print(csv_path)
print(jsonl_path)
print(f"总条数：{len(df)}")