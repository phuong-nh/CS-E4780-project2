import re


def postprocess_cypher(query: str) -> str:
    q = query.strip()

    # 1. remove any trailing semicolons
    q = re.sub(r";+\s*$", "", q)

    # 2. normalize whitespace
    q = re.sub(r"\s+", " ", q)

    # 3. convert exact matches into CONTAINS form
    # e.g.: p.category = 'physics' → toLower(p.category) CONTAINS 'physics'
    def convert_equals(m):
        left = m.group(1)
        value = m.group(2).lower()
        return f"toLower({left}) CONTAINS '{value}'"

    q = re.sub(r"(\w+\.\w+)\s*=\s*'([^']+)'", convert_equals, q)

    # 4. RETURN clause fixes
    # RETURN s → RETURN s.knownName
    q = re.sub(r"RETURN\s+([a-zA-Z][a-zA-Z0-9_]*)\s*$", r"RETURN \1.knownName", q)

    # 5. normalize whitespace again
    q = re.sub(r"\s+", " ", q).strip()

    return q
