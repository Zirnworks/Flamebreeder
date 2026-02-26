"""Genome representation for fractal flame breeding.

A Genome encapsulates a latent vector plus metadata for persistence
and lineage tracking.
"""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Genome:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latent_vector: list[float] = field(default_factory=list)
    latent_space: str = "w"  # "z" (FastGAN) or "w" (StyleGAN2)
    truncation_psi: float | None = None
    parents: tuple[str, str] | None = None
    breeding_method: str | None = None
    breeding_params: dict | None = None
    generation: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: list[str] = field(default_factory=list)
    favorite: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuple to list for JSON serialization
        if d["parents"] is not None:
            d["parents"] = list(d["parents"])
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Genome":
        if data.get("parents") is not None:
            data["parents"] = tuple(data["parents"])
        # Backward compat: old genomes without latent_space are Z-space
        if "latent_space" not in data:
            data["latent_space"] = "z"
        # Filter to known fields only
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls.from_dict(json.loads(s))


class GenomeStore:
    """Persistent storage for genomes as a JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._genomes: dict[str, Genome] = {}
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path) as f:
            data = json.load(f)
        for d in data:
            g = Genome.from_dict(d)
            self._genomes[g.id] = g

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([g.to_dict() for g in self._genomes.values()], f)

    def add(self, genome: Genome):
        self._genomes[genome.id] = genome
        self._save()

    def get(self, genome_id: str) -> Genome | None:
        return self._genomes.get(genome_id)

    def all(self) -> list[Genome]:
        return list(self._genomes.values())

    def delete(self, genome_id: str):
        self._genomes.pop(genome_id, None)
        self._save()

    def update(self, genome: Genome):
        self._genomes[genome.id] = genome
        self._save()
