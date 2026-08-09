"""Registry of knowledge bases: every subfolder under data/ is discovered automatically,
so adding a new knowledge base later means just dropping in a new data/<name>/ folder."""
from dataclasses import dataclass
from pathlib import Path

from bedrock_client import DATA_DIR

DESCRIPTION_FILE = "_description.txt"
INDEX_DIRNAME = "index"
FAISS_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


@dataclass(frozen=True)
class KnowledgeBase:
    name: str
    description: str
    source_dir: Path
    index_dir: Path

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / FAISS_FILENAME

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / CHUNKS_FILENAME


def get_registry() -> dict[str, KnowledgeBase]:
    """Scan data/ for subfolders and build the kb_name -> KnowledgeBase map."""
    if not DATA_DIR.is_dir():
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    registry = {}
    for entry in sorted(DATA_DIR.iterdir()):
        if not entry.is_dir():
            continue
        description_path = entry / DESCRIPTION_FILE
        description = (
            description_path.read_text(encoding="utf-8").strip()
            if description_path.is_file()
            else f"Knowledge base of documents about {entry.name.replace('-', ' ').replace('_', ' ')}."
        )
        registry[entry.name] = KnowledgeBase(
            name=entry.name,
            description=description,
            source_dir=entry,
            index_dir=entry / INDEX_DIRNAME,
        )
    return registry