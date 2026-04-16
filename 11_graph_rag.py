"""
Bài 11: GraphRAG — Knowledge Graph + LLM
Mục tiêu: Trích xuất entity/relation từ text, query qua graph
(Phiên bản đơn giản, không cần Neo4j)
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.1,  # Rất thấp cho extraction chính xác
)


# ============================
# BƯỚC 1: Entity Extraction từ text
# ============================
print("=" * 60)
print("📌 BƯỚC 1: ENTITY & RELATION EXTRACTION")
print("=" * 60)

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là hệ thống trích xuất knowledge graph từ văn bản.
Hãy trích xuất các ENTITY (thực thể) và RELATION (mối quan hệ) từ đoạn text.

Trả về dạng JSON với format:
{{
    "entities": [
        {{"name": "tên entity", "type": "PERSON/ORG/POLICY/NUMBER/TIME"}},
    ],
    "relations": [
        {{"source": "entity_1", "relation": "mối quan hệ", "target": "entity_2"}},
    ]
}}

CHỈ trả về JSON, không giải thích thêm."""),
    ("user", "{text}"),
])

extract_chain = extraction_prompt | llm | StrOutputParser()


# Đọc file
with open("data/company_docs.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# Trích xuất từ đoạn text nhỏ (để demo)
sample_sections = full_text.split("## ")[1:5]  # Lấy 4 sections đầu

knowledge_graph = {"entities": [], "relations": []}

for i, section in enumerate(sample_sections):
    section_text = section[:500]  # Giới hạn để tránh quá dài
    print(f"\n📝 Section {i+1}: {section_text[:50]}...")

    try:
        result = extract_chain.invoke({"text": section_text})

        # Clean JSON (loại bỏ markdown code block nếu có)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]

        parsed = json.loads(result)

        # Gộp vào knowledge graph
        knowledge_graph["entities"].extend(parsed.get("entities", []))
        knowledge_graph["relations"].extend(parsed.get("relations", []))

        print(f"  ✅ Entities: {len(parsed.get('entities', []))}")
        print(f"  ✅ Relations: {len(parsed.get('relations', []))}")

    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️ Parse error: {e}")


# ============================
# BƯỚC 2: Hiển thị Knowledge Graph
# ============================
print("\n\n" + "=" * 60)
print("📊 KNOWLEDGE GRAPH")
print("=" * 60)

# Unique entities
unique_entities = {}
for e in knowledge_graph["entities"]:
    name = e["name"].lower()
    if name not in unique_entities:
        unique_entities[name] = e["type"]

print(f"\n🔵 Entities ({len(unique_entities)}):")
for name, etype in unique_entities.items():
    print(f"  [{etype}] {name}")

print(f"\n🔗 Relations ({len(knowledge_graph['relations'])}):")
for r in knowledge_graph["relations"]:
    print(f"  {r['source']} --[{r['relation']}]--> {r['target']}")


# ============================
# BƯỚC 3: Graph-based Query
# ============================
print("\n\n" + "=" * 60)
print("🔍 GRAPH-BASED QUERY")
print("=" * 60)

# Tạo adjacency list từ relations
graph = {}
for r in knowledge_graph["relations"]:
    src = r["source"].lower()
    tgt = r["target"].lower()
    rel = r["relation"]

    if src not in graph:
        graph[src] = []
    graph[src].append({"relation": rel, "target": tgt})

    # Bidirectional
    if tgt not in graph:
        graph[tgt] = []
    graph[tgt].append({"relation": f"reverse_{rel}", "target": src})


def find_related_info(entity_name, graph, depth=2):
    """Tìm tất cả thông tin liên quan đến entity trong graph."""
    visited = set()
    results = []

    def dfs(node, current_depth, path):
        if current_depth > depth or node in visited:
            return
        visited.add(node)

        if node in graph:
            for edge in graph[node]:
                results.append({
                    "path": path + [f"--[{edge['relation']}]-->", edge["target"]],
                    "depth": current_depth,
                })
                dfs(edge["target"], current_depth + 1,
                    path + [f"--[{edge['relation']}]-->", edge["target"]])

    dfs(entity_name.lower(), 0, [entity_name])
    return results


# Query qua graph
print("\n❓ Query: 'Thông tin về nhân viên thử việc'")
related = find_related_info("thử việc", graph, depth=2)

if related:
    print(f"\n📊 Tìm thấy {len(related)} connections:")
    for r in related:
        path_str = " ".join(r["path"])
        print(f"  (depth {r['depth']}) {path_str}")
else:
    print("  Không tìm thấy entity trong graph.")
    print("  Thử entity khác từ danh sách entities ở trên.")


# ============================
# BƯỚC 4: GraphRAG — Kết hợp Graph + LLM
# ============================
print("\n\n" + "=" * 60)
print("🤖 GraphRAG — Graph Context + LLM")
print("=" * 60)

graphrag_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là trợ lý AI. Trả lời câu hỏi dựa trên Knowledge Graph context.

GRAPH_CONTEXT chứa các entities và relations trích xuất từ tài liệu công ty.
Sử dụng các mối quan hệ để trả lời chính xác."""),
    ("user", """KNOWLEDGE GRAPH CONTEXT:

Entities:
{entities}

Relations:
{relations}

CÂU HỎI: {question}
"""),
])

graphrag_chain = graphrag_prompt | llm | StrOutputParser()


def graphrag_query(question):
    """Query sử dụng knowledge graph context."""

    # Format entities
    entities_text = "\n".join(
        f"- [{etype}] {name}"
        for name, etype in unique_entities.items()
    )

    # Format relations
    relations_text = "\n".join(
        f"- {r['source']} --[{r['relation']}]--> {r['target']}"
        for r in knowledge_graph["relations"]
    )

    answer = graphrag_chain.invoke({
        "entities": entities_text,
        "relations": relations_text,
        "question": question,
    })
    return answer


questions = [
    "Quy trình phỏng vấn tại công ty gồm bao nhiêu bước?",
    "Chính sách bảo hiểm cho nhân viên và người thân?",
    "Career path cho lập trình viên từ junior đến senior?",
]

for q in questions:
    print(f"\n❓ {q}")
    answer = graphrag_query(q)
    print(f"✅ {answer}\n")
    print("-" * 50)
