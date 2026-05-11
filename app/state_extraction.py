# app/state_extraction.py
"""
Lightweight state extraction — supports retrieval and grounding only.
Conversational policy is the LLM's responsibility.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Retrieval-support snapshot of the conversation."""
    role_title: Optional[str] = None
    domain_keywords: list[str] = field(default_factory=list)
    seniority_text: Optional[str] = None
    language_required: Optional[str] = None
    wants_personality: bool = False
    wants_cognitive: bool = False
    wants_sjt: bool = False
    wants_simulation: bool = False
    wants_remote: bool = False
    duration_budget: Optional[int] = None
    compare_targets: list[str] = field(default_factory=list)
    user_confirmed_final: bool = False
    off_topic: bool = False
    hiring_stage: Optional[str] = None
    deployment_goal: Optional[str] = None

    @property
    def desired_keys(self) -> list[str]:
        keys: list[str] = []
        if self.wants_personality:
            keys.append("Personality & Behavior")
        if self.wants_cognitive:
            keys.append("Ability & Aptitude")
        if self.wants_sjt:
            keys.append("Biodata & Situational Judgment")
        if self.wants_simulation:
            keys.append("Simulations")
        return keys

    def build_query_string(self) -> str:
        parts: list[str] = []
        if self.role_title:
            parts.append(self.role_title)
        if self.domain_keywords:
            parts.append(" ".join(self.domain_keywords[:10]))
        if self.seniority_text:
            parts.append(self.seniority_text)
        if self.language_required:
            parts.append(self.language_required)
        if self.desired_keys:
            parts.append(" ".join(self.desired_keys))
        if self.hiring_stage == "development":
            parts.append("development report coaching")
        elif self.hiring_stage in ("audit", "reskilling"):
            parts.append("skills audit benchmark development")
        return " ".join(parts).lower()

    def has_enough_info(self) -> bool:
        has_role = bool(self.role_title) or len(self.domain_keywords) >= 2
        has_constraint = bool(
            self.seniority_text or self.wants_personality or self.wants_cognitive
            or self.wants_sjt or self.wants_simulation
            or self.duration_budget is not None or self.language_required
        )
        return has_role and has_constraint

    def recommendation_confidence(self) -> float:
        score = 0.0
        if self.role_title:
            score += 0.25
        if len(self.domain_keywords) >= 2:
            score += 0.15
        if self.seniority_text:
            score += 0.20
        if any([self.wants_personality, self.wants_cognitive, self.wants_sjt, self.wants_simulation]):
            score += 0.20
        if self.language_required:
            score += 0.10
        if self.duration_budget:
            score += 0.10
        return min(1.0, score)


_SENIORITY_RE = re.compile(
    r"\b(junior|senior|mid[- ]?level|entry[- ]?level|graduate|intern|trainee"
    r"|manager|director|executive|cxo|ceo|cfo|cto|supervisor|lead"
    r"|experienced|beginner|fresher|fresh)\b", re.IGNORECASE,
)
_DURATION_RE = re.compile(r"\b(\d{1,3})\s*(?:min(?:ute)?s?|mins?)\b", re.IGNORECASE)
_QUICK_SCREEN_RE = re.compile(
    r"\b(quickly?\s+screen|short\s+(?:test|assessment)|brief\s+assessment|rapid\s+screen)\b",
    re.IGNORECASE,
)
_LANGUAGE_RE = re.compile(
    r"\b(?:in\s+|spoken\s+)?"
    r"(english(?:\s*\(?\s*(?:us|usa|uk|international)\s*\)?)?|spanish|french"
    r"|german|portuguese|chinese|japanese|dutch|arabic|hindi|mandarin)\b",
    re.IGNORECASE,
)
_PERSONALITY_RE = re.compile(
    r"\b(personality|behavio(?:u?r)|opq|trait|temperament"
    r"|motivation\s+questionnaire|competency\s+report|leadership\s+report|360)\b",
    re.IGNORECASE,
)
_COGNITIVE_RE = re.compile(
    r"\b(cognitive|aptitude|reasoning|numerical|verbal|inductive|deductive"
    r"|abstract|verify|g\+|logical|analytical|problem[- ]?solving)\b",
    re.IGNORECASE,
)
_SJT_RE = re.compile(r"\b(sjt|situational\s+judg(?:e?ment)|biodata|scenarios)\b", re.IGNORECASE)
_SIMULATION_RE = re.compile(
    r"\b(simulation|role[- ]?play|inbox|in[- ]?tray|call\s+simulation"
    r"|phone\s+simulation|case\s+study)\b", re.IGNORECASE,
)
_REMOTE_RE = re.compile(r"\b(remote|online|virtual|proctored\s+remotely|unproctored)\b", re.IGNORECASE)
_OFF_TOPIC_RE = re.compile(
    r"\b(legal\s+(?:require|complian|advi|risk|liability)|labor\s+law"
    r"|employment\s+law|discrimination\s+law|gdpr|hipaa|eeoc|ada\s+complian"
    r"|hiring\s+(?:strategy|best\s+practice|tip|guide)"
    r"|salary\s+negotiat|compensation\s+structure|offer\s+letter)\b",
    re.IGNORECASE,
)
_INJECTION_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:prior|previous|above)\s+(?:instruction|prompt|rule)"
    r"|forget\s+(?:your|all)\s+(?:instruction|rule|prompt)"
    r"|you\s+are\s+now\s+(?:a|an)\s+(?!shl)"
    r"|disregard\s+(?:your|all|the)\s+(?:system|instruction))",
    re.IGNORECASE,
)
_NON_SHL_RE = re.compile(
    r"\b(disc\s+assessment|myers[- ]?briggs|mbti|gallup\s+strengths"
    r"|strengthsfinder|hogan\s+assessment|caliper|predictive\s+index"
    r"|wonderlic|kolbe|enneagram)\b", re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"\b(differ(?:ence|ent|s)?(?:\s+between)?|compar(?:e|ing|ison)"
    r"|versus|vs\.?|how\s+(?:does|do|is|are)\s+\S+\s+(?:differ|compare))\b",
    re.IGNORECASE,
)
_CONFIRMED_RE = re.compile(
    r"\b((?:that(?:'s|\s+is)\s+)?(?:perfect|great|enough|fine|excellent)"
    r"|thanks?,?\s+(?:that|this)(?:\s+is\s+(?:helpful|what\s+i\s+needed))?"
    r"|(?:i(?:'m|\s+am)\s+)?(?:satisfied|happy)\s+with"
    r"|no\s+(?:more|further)\s+(?:questions?|changes?)"
    r"|looks?\s+good|that\s+works|we(?:'re|\s+are)\s+good"
    r"|confirmed|locking\s+it\s+in)\b",
    re.IGNORECASE,
)
_HIRING_STAGE_RE = re.compile(
    r"\b(screening|shortlisting|finalist|final\s+(?:stage|round|candidates)"
    r"|talent\s+audit|annual\s+audit|reskill(?:ing)?|development|coaching"
    r"|graduate\s+(?:intake|scheme)|trainee\s+scheme|selection)\b",
    re.IGNORECASE,
)
_DEPLOYMENT_GOAL_RE = re.compile(
    r"\b(benchmark(?:ing)?|identify\s+(?:potential|talent|high[- ]?pot)"
    r"|talent\s+audit|skills?\s+gap|reskill)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:remove|drop|exclude|no|don'?t\s+include|without|skip)\s+", re.IGNORECASE
)
_STOPWORDS: set[str] = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were",
    "can", "you", "need", "want", "looking", "find", "test", "tests",
    "assessment", "assessments", "please", "would", "like", "could",
    "about", "from", "have", "has", "our", "their", "some", "your",
    "should", "also", "any", "will", "not", "but", "than", "more",
    "into", "been", "being", "what", "which", "who", "how", "why",
    "remote", "online", "minutes", "min", "mins", "minute",
    "use", "using", "recommend", "suggestion", "suggest", "help",
    "good", "best", "right", "suitable", "appropriate", "there",
    "does", "don", "don't", "its", "they", "them", "then", "when",
    "where", "here", "just", "very", "really", "actually", "add",
    "remove", "drop", "shl", "product", "products", "catalog",
    "solution", "solutions", "people", "person", "candidate",
    "candidates", "role", "roles", "position", "hire", "hiring",
    "level", "levels", "type", "types", "team", "staff", "employee",
    "company", "work", "working", "job", "new", "current", "we",
    "us", "me", "my",
}


def extract_state_from_messages(messages: list[ChatMessage]) -> ConversationState:
    state = ConversationState()
    user_texts = [m.content for m in messages if m.role.value == "user"]
    all_text = " ".join(user_texts)
    last_text = user_texts[-1] if user_texts else ""

    if _OFF_TOPIC_RE.search(last_text) or _INJECTION_RE.search(last_text) or _NON_SHL_RE.search(last_text):
        state.off_topic = True

    m = _SENIORITY_RE.search(all_text)
    if m:
        state.seniority_text = m.group(1).lower()

    m = _DURATION_RE.search(all_text)
    if m:
        state.duration_budget = int(m.group(1))
    elif _QUICK_SCREEN_RE.search(all_text):
        state.duration_budget = 20

    m = _LANGUAGE_RE.search(all_text)
    if m:
        state.language_required = _normalize_language(m.group(1))

    if _REMOTE_RE.search(all_text):
        state.wants_remote = True
    if _PERSONALITY_RE.search(all_text):
        state.wants_personality = True
    if _COGNITIVE_RE.search(all_text):
        state.wants_cognitive = True
    if _SJT_RE.search(all_text):
        state.wants_sjt = True
    if _SIMULATION_RE.search(all_text):
        state.wants_simulation = True
    if re.search(r"\bspoken\s+english\b", all_text, re.IGNORECASE):
        state.wants_simulation = True

    m = _HIRING_STAGE_RE.search(all_text)
    if m:
        state.hiring_stage = _norm_stage(m.group(1).lower())
    m = _DEPLOYMENT_GOAL_RE.search(all_text)
    if m:
        state.deployment_goal = _norm_goal(m.group(1).lower())

    if _COMPARE_RE.search(last_text):
        state.compare_targets = _extract_product_names(last_text)

    if _CONFIRMED_RE.search(last_text):
        state.user_confirmed_final = True

    words = re.findall(r"[a-zA-Z+#.]{2,}", all_text.lower())
    state.domain_keywords = [w for w in dict.fromkeys(words) if w not in _STOPWORDS][:15]

    if user_texts:
        first = user_texts[0].strip()
        if len(first.split()) <= 30:
            state.role_title = first

    _apply_negations(state, last_text)
    return state


def _normalize_language(raw: str) -> str:
    low = raw.lower().strip()
    if low.startswith("english"):
        if "us" in low or "usa" in low:
            return "English (USA)"
        if "uk" in low:
            return "English (UK)"
        if "international" in low:
            return "English (International)"
        return "English"
    return raw.strip().title()


def _norm_stage(raw: str) -> str:
    if any(k in raw for k in ("screen", "shortlist")):
        return "screening"
    if any(k in raw for k in ("finalist", "final")):
        return "finalist"
    if "reskill" in raw:
        return "reskilling"
    if "audit" in raw:
        return "audit"
    if "development" in raw or "coaching" in raw:
        return "development"
    if any(k in raw for k in ("graduate", "trainee")):
        return "graduate_intake"
    return "selection"


def _norm_goal(raw: str) -> str:
    if "benchmark" in raw:
        return "benchmarking"
    if any(k in raw for k in ("develop", "coaching", "skill", "reskill")):
        return "development"
    if any(k in raw for k in ("audit", "potential", "high-pot")):
        return "talent_audit"
    return "selection"


def _extract_product_names(text: str) -> list[str]:
    names: list[str] = []
    quoted = re.findall(r'["\']([^"\']{3,})["\']', text)
    names.extend(quoted)
    known = re.findall(
        r"\b(OPQ\s*\S*(?:\s+\S+){0,3}|Verify\s+\S+(?:\s+\S+){0,2}"
        r"|MQ\s+\S+(?:\s+\S+){0,2}|DSI\b"
        r"|Universal\s+Competency\s+\S+(?:\s+\S+){0,2}"
        r"|Safety\s+(?:&|and)\s+Dependability\s+\S+"
        r"|Contact\s+Center\s+\S+(?:\s+\S+){0,3}"
        r"|Graduate\s+Scenarios)",
        text, re.IGNORECASE,
    )
    names.extend(known)
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        nc = n.strip()
        if nc.lower() not in seen and len(nc) > 2:
            seen.add(nc.lower())
            unique.append(nc)
    return unique


def _apply_negations(state: ConversationState, text: str) -> None:
    low = text.lower()
    if not _NEGATION_RE.search(low):
        return
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:personality|opq|behavio)", low):
        state.wants_personality = False
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:cognitive|aptitude|reasoning|verify)", low):
        state.wants_cognitive = False
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:sjt|situational|judgment)", low):
        state.wants_sjt = False
    if re.search(r"\b(?:remove|drop|no|without|skip|exclude|don'?t\s+include)\s+(?:.*?)(?:simulation|role.?play|inbox)", low):
        state.wants_simulation = False


def merge_state(old: ConversationState, new: ConversationState) -> ConversationState:
    return ConversationState(
        role_title=new.role_title or old.role_title,
        domain_keywords=new.domain_keywords or old.domain_keywords,
        seniority_text=new.seniority_text or old.seniority_text,
        language_required=new.language_required or old.language_required,
        wants_personality=new.wants_personality or old.wants_personality,
        wants_cognitive=new.wants_cognitive or old.wants_cognitive,
        wants_sjt=new.wants_sjt or old.wants_sjt,
        wants_simulation=new.wants_simulation or old.wants_simulation,
        wants_remote=new.wants_remote or old.wants_remote,
        duration_budget=(new.duration_budget if new.duration_budget is not None else old.duration_budget),
        compare_targets=new.compare_targets or old.compare_targets,
        user_confirmed_final=new.user_confirmed_final,
        off_topic=new.off_topic,
        hiring_stage=new.hiring_stage or old.hiring_stage,
        deployment_goal=new.deployment_goal or old.deployment_goal,
    )