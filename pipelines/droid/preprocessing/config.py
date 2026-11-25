from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class EpisodeSelection:
    episode_id: str
    source_relative_path: str

    @property
    def source_folder_name(self) -> str:
        return self.source_relative_path.rstrip("/").split("/")[-1]
