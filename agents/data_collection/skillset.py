from functools import lru_cache
from pathlib import Path


SKILLS_DIR = Path(__file__).with_name("skills")


@lru_cache(maxsize=None)
def load_skill(name: str) -> str:
    skill_path = SKILLS_DIR / name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Unknown skill '{name}' at {skill_path}.")
    content = skill_path.read_text(encoding="utf-8")
    if content.startswith("---\n"):
        _, _, remainder = content.split("---\n", 2)
        return remainder.lstrip()
    return content
