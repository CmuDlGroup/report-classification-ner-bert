# FastAPI endpoint for Aviation ADREP Classification
# Pipeline: narrative -> NER (SafeAeroBERT) -> entity extraction -> ADREP scoring -> API response

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
from collections import defaultdict

app = FastAPI()

MODEL_ID = "theophilusowiti/asn-ner-aerobert"
MODEL_DISPLAY_NAME = "SafeAeroBERT NER + ADREP Classifier"

# Keywords matched by substring against extracted entity text.
# TRIGGER entities are weighted 3x, so list the most discriminative TRIGGER phrases first.
ADREP_KEYWORDS: dict = {
    "SCF-PP": [
        "engine failure", "engine fire", "engine malfunction", "engine problem",
        "engine separation", "engine shutdown", "engine surge", "power loss",
        "flameout", "oil leak", "fuel leak", "compressor stall", "turbine failure",
        "propeller failure", "rpm rollback", "powerplant", "turbine", "compressor",
        "propeller", "engine",
    ],
    "SCF-NP": [
        "gear failure", "gear collapse", "nose gear", "main gear", "landing gear",
        "hydraulic failure", "hydraulic leak", "avionics failure", "flap failure",
        "flight control failure", "structural failure", "electrical failure",
        "hydraulic", "avionics", "flap", "rudder", "elevator", "aileron",
    ],
    "RE": [
        "runway excursion", "runway overrun", "overran runway", "veered off runway",
        "skidded off", "directional control", "overrun", "excursion",
    ],
    "LOC-I": [
        "loss of control", "departure from controlled flight", "unusual attitude",
        "uncontrolled descent", "spiral dive", "pitch up", "stall", "upset", "spin",
    ],
    "CFIT": [
        "controlled flight into terrain", "struck trees", "struck high ground",
        "hit terrain", "terrain impact", "ground impact", "terrain", "mountain",
        "hill", "tree",
    ],
    "ARC": [
        "hard landing", "tail strike", "tailstrike", "nose gear collapse",
        "gear up landing", "bounced", "firm touchdown", "rough landing",
    ],
    "TURB": [
        "severe turbulence", "clear air turbulence", "wake turbulence",
        "turbulence", "chop", "jolt",
    ],
    "WSTRW": [
        "microburst", "downburst", "windshear", "wind shear", "shear",
    ],
    "ICE": [
        "ice accretion", "ice ingestion", "icing", "frost", "frozen",
        "deice", "anti-ice", "ice",
    ],
    "FUEL": [
        "fuel exhaustion", "fuel starvation", "fuel contamination", "fuel imbalance",
        "low fuel", "exhaustion", "starvation", "fuel",
    ],
    "FIRE": [
        "in-flight fire", "cabin fire", "cargo fire", "electrical fire",
        "smoke", "fumes", "fire",
    ],
    "WILD": [
        "bird strike", "bird ingestion", "wildlife strike", "animal strike",
        "bird", "wildlife", "animal",
    ],
    "MAC": [
        "mid-air collision", "midair collision", "airprox", "near miss", "tcas",
        "traffic alert", "collision",
    ],
    "GCOL": [
        "ground collision", "taxiway collision", "ramp collision", "tug", "pushback",
    ],
    "SEC": [
        "hijack", "hijacking", "air piracy", "security threat", "bomb threat",
        "weapon", "unruly passenger", "assault", "attack",
    ],
    "UIMC": [
        "inadvertent imc", "vfr into imc", "flew into cloud",
        "instrument meteorological conditions", "imc",
    ],
    "OTHR": [],
}

# Entity type scoring weights — TRIGGERs are primary accident drivers
ENTITY_WEIGHTS = {
    "TRIGGER": 3.0,
    "OUTCOME": 2.0,
    "SYSTEM":  1.5,
    "PHASE":   0.5,
    "ACTOR":   0.5,
}

ALL_ADREP_CODES = list(ADREP_KEYWORDS.keys())

print("Loading NER model...")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
_model.eval()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model.to(_device)
print(f"Model loaded on {_device}.")


class IncidentRequest(BaseModel):
    narrative: str
    event_id: str = None


def extract_entities(text: str) -> list:
    """Run NER inference; return [(token, label), ...] with subwords merged."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []
    for token, label_id in zip(tokens, preds):
        if token in _tokenizer.all_special_tokens:
            continue
        label = _model.config.id2label[label_id]
        if token.startswith("##"):
            if results:
                results[-1] = (results[-1][0] + token[2:], results[-1][1])
        else:
            results.append((token, label))

    return results


def build_event_dict(entities: list) -> dict:
    """
    Merge consecutive B-/I- tokens into multi-word phrases.
    Returns {"ACTOR": [...], "SYSTEM": [...], "PHASE": [...],
             "TRIGGER": [...], "OUTCOME": [...]}
    """
    event = {role: [] for role in ENTITY_WEIGHTS}
    current_tokens = []
    current_role = None

    for token, label in entities:
        if label == "O":
            if current_tokens and current_role:
                event[current_role].append(" ".join(current_tokens))
            current_tokens, current_role = [], None
            continue

        prefix, role = label.split("-", 1)
        if role not in event:
            continue

        if prefix == "B":
            if current_tokens and current_role:
                event[current_role].append(" ".join(current_tokens))
            current_tokens = [token]
            current_role = role
        elif prefix == "I" and role == current_role:
            current_tokens.append(token)
        else:
            if current_tokens and current_role:
                event[current_role].append(" ".join(current_tokens))
            current_tokens = [token]
            current_role = role

    if current_tokens and current_role:
        event[current_role].append(" ".join(current_tokens))

    return event


def score_adrep(event: dict) -> dict:
    """
    For each entity role, check which ADREP codes have a keyword that is a
    substring of the extracted phrase. Accumulate weighted scores.
    """
    scores: dict = defaultdict(float)

    for role, phrases in event.items():
        weight = ENTITY_WEIGHTS.get(role, 1.0)
        combined = " ".join(phrases).lower()
        for code, keywords in ADREP_KEYWORDS.items():
            for kw in keywords:
                if kw in combined:
                    scores[code] += weight
                    break  # count each code once per entity type

    if not scores:
        scores["OTHR"] = 1.0

    return dict(scores)


def scores_to_top5(scores: dict) -> tuple:
    """Normalise scores → confidences; return (top_class, confidence, top_5)."""
    total = sum(scores.values())
    normalised = {k: v / total for k, v in scores.items()}

    # Assign a tiny residual to every code not already scored
    unscored = [c for c in ALL_ADREP_CODES if c not in normalised]
    residual = max((1.0 - sum(normalised.values())) / max(len(unscored), 1), 0.001)
    for code in unscored:
        normalised[code] = residual

    sorted_codes = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
    top_class, top_conf = sorted_codes[0]
    top_5 = [{"class": c, "confidence": round(conf, 4)} for c, conf in sorted_codes[:5]]

    return top_class, round(top_conf, 4), top_5


@app.post("/predict")
async def predict(request: IncidentRequest):
    start = time.time()

    entities = extract_entities(request.narrative)
    event = build_event_dict(entities)
    scores = score_adrep(event)
    top_class, confidence, top_5 = scores_to_top5(scores)

    return {
        "model_id": MODEL_ID,
        "display_name": MODEL_DISPLAY_NAME,
        "prediction": {
            "top_class": top_class,
            "confidence": confidence,
            "top_5": top_5,
        },
        "inference_time_ms": int((time.time() - start) * 1000),
    }
