from __future__ import annotations

import json

from pydantic.dataclasses import dataclass


@dataclass
class Patch:
    """Annotated image patch metadata.

    Args:
        file: filename of the associated image. Note: This is not a path
        coords: front lower left vertex of patch
        size: size in pixels in each direction
    """
    file: str
    coords: tuple[int, int, int]
    size: tuple[int, int, int]

    @classmethod
    def from_json(cls, data: str) -> Patch | list[Patch]:
        """Create Patch or list of Patch object from json string.

        Args:
            data: json string of a patch or a list of patches

        Returns:
            patch object or a list of patch objects
        """
        parsed = json.loads(data)
        match parsed:
            case [*_]:
                return [Patch(**item) for item in parsed]
            case {}:
                return Patch(**parsed)
            case _:
                raise NotImplementedError
