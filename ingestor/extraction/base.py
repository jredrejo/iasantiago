"""
Clases base y tipos para la capa de extracción.

Proporciona la dataclass Element para representación unificada de elementos
y la interfaz ExtractorProtocol para implementaciones de extracción.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple


class ExtractionError(Exception):
    """Se lanza cuando todos los métodos de extracción fallan."""

    pass


@dataclass
class Element:
    """
    Representación unificada de un elemento de documento extraído.

    Reemplaza diccionarios ad-hoc con una estructura tipada para
    manejo consistente en todo el pipeline de extracción.
    """

    text: str
    type: str  # 'text', 'table', 'image', 'title', etc.
    page: int
    source: str  # 'pypdf', 'pdfplumber', 'docling', 'easyocr', etc.
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para compatibilidad hacia atrás."""
        result = {
            "text": self.text,
            "type": self.type,
            "page": self.page,
            "source": self.source,
        }
        if self.bbox is not None:
            result["bbox"] = self.bbox
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Element":
        """Crea un Element desde un diccionario."""
        return cls(
            text=str(data.get("text", "")),
            type=str(data.get("type", "text")),
            page=int(data.get("page", 1)),
            source=str(data.get("source", "unknown")),
            bbox=data.get("bbox"),
            metadata=data.get("metadata", {}),
        )


class ExtractorProtocol(Protocol):
    """
    Interfaz para todas las implementaciones de extracción de PDF.

    Los extractores deben implementar:
    - extract(): Método principal de extracción que retorna una lista de elementos
    - can_handle(): Verifica si este extractor puede procesar el archivo dado
    - name: Nombre legible para logging
    """

    @property
    def name(self) -> str:
        """Nombre legible para este extractor."""
        ...

    def extract(self, pdf_path: Path) -> List[Element]:
        """
        Extrae elementos de un archivo PDF.

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Lista de objetos Element extraídos

        Raises:
            ExtractionError: Si la extracción falla
        """
        ...

    def can_handle(self, pdf_path: Path) -> bool:
        """
        Verifica si este extractor puede manejar el archivo dado.

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            True si este extractor puede procesar el archivo
        """
        ...


def elements_to_dicts(elements: List[Element]) -> List[Dict[str, Any]]:
    """Convierte una lista de Elements a diccionarios para compatibilidad hacia atrás."""
    return [e.to_dict() for e in elements]


def dicts_to_elements(dicts: List[Dict[str, Any]]) -> List[Element]:
    """Convierte una lista de diccionarios a Elements."""
    return [Element.from_dict(d) for d in dicts]


def check_pdf_has_text(pdf_path: Path, min_chars: int = 100) -> bool:
    """
    Verifica si un PDF tiene capa de texto extraíble.

    Args:
        pdf_path: Ruta al archivo PDF
        min_chars: Mínimo de caracteres para considerar texto presente

    Returns:
        True si el PDF tiene capa de texto, False si es escaneado/solo imagen
    """
    try:
        import pypdf

        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            total_text = 0
            # Verificar primeras páginas
            for page in reader.pages[:5]:
                text = page.extract_text() or ""
                total_text += len(text.strip())
                if total_text >= min_chars:
                    return True
        return False
    except Exception:
        return False
