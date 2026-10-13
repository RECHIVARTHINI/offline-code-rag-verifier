import httpx
import json
import re
from typing import Dict, Any, List

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:1.5b"

PROMPT_TEMPLATE = """You are an Offline Zero-Trust Code Understanding Engine.
Answer the user question accurately using ONLY the provided code chunks.
DO NOT invent or guess any function names, parameters, or call relationships.

Context Chunks:
{context_chunks}

User Query:
{query}

You MUST follow this exact response structure:
1. Provide a concise technical answer in markdown. Every factual sentence in your technical answer must map directly to a verified claim in the structured summary below.
2. At the very end of your response, output a structured JSON summary inside <claims>...</claims> tags.
IMPORTANT:
- Every factual sentence in your technical answer MUST map to a claim in "factual_assertions", "symbols_mentioned", or "call_edges".
- For "symbols_mentioned" and "structures", copy the EXACT FQN strings from the Chunk headers above (e.g. "routes/analyticsRoutes.js::GET /:crop" or "public/script.js::createCropCard").

Example of the mandatory ending block:
<claims>
```json
{{
  "symbols_mentioned": ["routes/paymentRoutes.js::POST /create-order"],
  "structures": [
    {{"symbol": "routes/paymentRoutes.js::POST /create-order", "claimed_params": ["req", "res"], "claimed_parent": "router"}}
  ],
  "call_edges": [
    {{"caller": "routes/paymentRoutes.js::POST /create-order", "callee": "cashfreeRequest"}}
  ],
  "factual_assertions": [
    "Asserts that the POST /create-order route creates a payment order."
  ]
}}
```
</claims>
"""

async def generate_code_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    formatted_context = ""
    for i, c in enumerate(context_chunks):
        point = c.get("point")
        payload = getattr(point, "payload", {}) if point else c.get("payload", c)
        file_path = payload.get("file_path", "unknown")
        symbol = payload.get("symbol", "unknown")
        fqn = payload.get("fqn", f"{file_path}::{symbol}")
        content = payload.get("content", "")
        formatted_context += f"--- Chunk {i+1} [FQN: {fqn}] ---\n{content}\n\n"

    prompt = PROMPT_TEMPLATE.format(context_chunks=formatted_context, query=query)
    
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 1200
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            raw_text = result.get("response", "")
        except Exception as e:
            return {
                "explanation": f"Error communicating with local Ollama ({OLLAMA_MODEL}): {str(e)}",
                "claims": {},
                "raw_response": "",
                "error": str(e)
            }

    claims_data = {}
    explanation = raw_text

    claims_match = re.search(r'<claims>\s*```(?:json)?(.*?)```\s*</claims>', raw_text, re.DOTALL)
    if claims_match:
        try:
            claims_data = json.loads(claims_match.group(1).strip())
            explanation = raw_text[:claims_match.start()].strip()
        except Exception:
            claims_data = {}
    else:
        json_block = re.search(r'```json(.*?)```', raw_text, re.DOTALL)
        if json_block:
            try:
                claims_data = json.loads(json_block.group(1).strip())
                explanation = raw_text[:json_block.start()].strip()
            except Exception:
                pass

        # Try to parse truncated/unclosed JSON block if present
        if not claims_data and "<claims>" in raw_text:
            after_claims = raw_text.split("<claims>", 1)[1]
            explanation = raw_text.split("<claims>", 1)[0].strip()
            # Extract JSON substring
            json_text_match = re.search(r'\{.*', after_claims, re.DOTALL)
            if json_text_match:
                candidate_json = json_text_match.group(0).strip()
                # Try closing open structures
                for suffix in ["\n}", "\n]}", "\n\"]}", "\n\"]}}", "\n]}}"]:
                    try:
                        claims_data = json.loads(candidate_json + suffix)
                        if isinstance(claims_data, dict):
                            break
                    except Exception:
                        continue
        elif not claims_data and "```json" in raw_text:
            after_json = raw_text.split("```json", 1)[1]
            explanation = raw_text.split("```json", 1)[0].strip()
            json_text_match = re.search(r'\{.*', after_json, re.DOTALL)
            if json_text_match:
                candidate_json = json_text_match.group(0).strip()
                for suffix in ["\n}", "\n]}", "\n\"]}", "\n\"]}}", "\n]}}"]:
                    try:
                        claims_data = json.loads(candidate_json + suffix)
                        if isinstance(claims_data, dict):
                            break
                    except Exception:
                        continue

    if not claims_data or not isinstance(claims_data, dict):
        found_symbols = set()
        backticked = re.findall(r'`([a-zA-Z0-9_/\\.:-]+)`', explanation)
        for b in backticked:
            found_symbols.add(b.strip())

        func_calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', explanation)
        for f in func_calls:
            if f not in ["if", "for", "while", "return", "function", "require", "import"]:
                found_symbols.add(f)

        sentences = [
            re.sub(r'^[#\*\-\d\.\s]+', '', s).strip()
            for s in re.split(r'[\.\n]+', explanation)
            if len(s.strip()) > 20 and not s.strip().startswith(("#", "```", "<"))
        ]
        
        claims_data = {
            "symbols_mentioned": list(found_symbols),
            "structures": [],
            "call_edges": [],
            "factual_assertions": sentences[:4]
        }

    return {
        "explanation": explanation,
        "claims": claims_data,
        "raw_response": raw_text
    }
