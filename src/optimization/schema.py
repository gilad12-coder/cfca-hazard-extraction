from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import FIELD_SPECS

NULL_TOKENS = {"", None, "null"}
SYMPTOM_FIELD_NAMES = ("symptom1", "symptom2", "symptom3")
EnumT = TypeVar("EnumT", bound=Enum)


def _coerce_optional_enum(
    value: object, enum_cls: Type[EnumT]
) -> Optional[EnumT]:
    """Normalize literals such as 'null' into Optional[Enum].

    Args:
        value: Candidate literal or enum instance.
        enum_cls: Enum type to coerce into.

    Returns:
        Optional[EnumT]: Enum instance or ``None`` when the value represents a null sentinel.
    """
    if value in NULL_TOKENS:
        return None
    if value is None or isinstance(value, enum_cls):
        return value
    return enum_cls(str(value))


def _coerce_optional_int_enum(
    value: object, enum_cls: Type[IntEnum]
) -> Optional[IntEnum]:
    """Normalize numeric enums that arrive as strings.

    Args:
        value: Literal that may represent an integer enum.
        enum_cls: IntEnum subclass to coerce into.

    Returns:
        Optional[IntEnum]: Enum instance or ``None`` when value is null-like.
    """
    if value in NULL_TOKENS:
        return None
    if value is None or isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        return enum_cls(int(value))
    return enum_cls(int(value))


def _null_to_none(value: object) -> Optional[str]:
    """Convert schema 'null' sentinel values to None.

    Args:
        value: String or literal possibly representing null.

    Returns:
        Optional[str]: ``None`` for null-like values, otherwise the string representation.
    """
    if value in NULL_TOKENS:
        return None
    if value is None:
        return None
    return str(value)


def _is_null_like(value: object) -> bool:
    """Return True when the value represents a null sentinel."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return True
    return False


def _field_description(name: str) -> str:
    """Return the human-readable description for the requested field.

    Args:
        name: Field identifier present in FIELD_SPECS.

    Returns:
        str: Description string from the field spec.
    """
    spec = FIELD_SPECS.get(name, {})
    return str(spec.get("description", ""))


class ReportType(str, Enum):
    """High-level hazard class selected by the reporter."""

    ODOR = "מפגע ריח"
    WASTE = "מפגע פסולת"
    SMOKE = "מפגע עשן"


class SmellType(str, Enum):
    """Odor family for nuisance smell reports."""

    FIRE = "שריפה"
    WOOD_STOVE = "קמין עצים"
    CHEMICAL = "כימי"
    SEWAGE = "שפכים"
    AGRICULTURE = "חקלאי"
    OTHER = "אחר"


class BurningMaterial(str, Enum):
    """Dominant material that is burning in smell incidents."""

    PLASTIC = "פלסטיק/ניילון"
    TIRES = "צמיגים"
    CHEMICALS = "כימיקלים"
    BRUSH = "גזם"
    UNKNOWN = "לא ידוע"


class SmellIntensity(IntEnum):
    """Ordinal odor intensity scale."""

    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6


class SymptomValue(str, Enum):
    """Canonical list of physiological symptoms."""

    BREATHING_DIFFICULTY = "קשיי נשימה"
    EYE_BURNING = "צריבה בעיניים"
    SKIN_IRRITATION = "גירוי בעור"
    DIZZINESS = "סחרחורת"
    HEADACHE = "כאב ראש"
    NAUSEA = "בחילה"
    THROAT_IRRITATION = "כאב או גירוי בגרון"
    OTHER = "אחר"


class SmokeColor(str, Enum):
    """Visual smoke color classes for waste/smoke hazards."""

    BLACK = "שחור"
    GRAY = "אפור"
    WHITE = "לבן"
    NONE = "אין עשן"


class ReporterLocation(str, Enum):
    """Reporter position relative to the nuisance source."""

    SOURCE = "במקום המפגע"
    FAR = "רחוק מהמפגע"


class EventState(str, Enum):
    """Binary state describing whether the event ended."""

    ONGOING = "0"
    ENDED = "1"


class HazardReport(BaseModel):
    """Aggregate hazard schema that embeds the unified symptom list."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    report_type: Optional[ReportType] = Field(
        default=None, alias="reportType", description=_field_description("reportType")
    )
    smell_type: Optional[SmellType] = Field(
        default=None, alias="smellType", description=_field_description("smellType")
    )
    burning_material: Optional[BurningMaterial] = Field(
        default=None,
        alias="burningMaterial",
        description=_field_description("burningMaterial"),
    )
    smell_intensity: Optional[SmellIntensity] = Field(
        default=None,
        alias="smellIntensity",
        description=_field_description("smellIntensity"),
    )
    symptoms: list[SymptomValue] = Field(
        default_factory=list,
        alias="symptoms",
        description=_field_description("symptoms"),
    )
    location: Optional[str] = Field(
        default=None, alias="location", description=_field_description("location")
    )
    smoke_color: Optional[SmokeColor] = Field(
        default=None, alias="smokeColor", description=_field_description("smokeColor")
    )
    comments: Optional[str] = Field(
        default=None, alias="comments", description=_field_description("comments")
    )
    reporter_location: Optional[ReporterLocation] = Field(
        default=None,
        alias="reporterLocation",
        description=_field_description("reporterLocation"),
    )
    event_ended: Optional[EventState] = Field(
        default=None, alias="eventEnded", description=_field_description("eventEnded")
    )
    event_date: Optional[str] = Field(
        default=None, alias="eventDate", description=_field_description("eventDate")
    )
    event_time: Optional[str] = Field(
        default=None, alias="eventTime", description=_field_description("eventTime")
    )
    reporter_name: Optional[str] = Field(
        default=None,
        alias="reporterName",
        description=_field_description("reporterName"),
    )
    home_address: Optional[str] = Field(
        default=None, alias="homeAddress", description=_field_description("homeAddress")
    )

    @field_validator("report_type", mode="before")
    @classmethod
    def _normalize_report_type(cls, value: object) -> Optional[ReportType]:
        return _coerce_optional_enum(value, ReportType)

    @field_validator("smell_type", mode="before")
    @classmethod
    def _normalize_smell_type(cls, value: object) -> Optional[SmellType]:
        return _coerce_optional_enum(value, SmellType)

    @field_validator("burning_material", mode="before")
    @classmethod
    def _normalize_burning_material(
        cls, value: object
    ) -> Optional[BurningMaterial]:
        return _coerce_optional_enum(value, BurningMaterial)

    @field_validator("smell_intensity", mode="before")
    @classmethod
    def _normalize_smell_intensity(cls, value: object) -> Optional[SmellIntensity]:
        return _coerce_optional_int_enum(value, SmellIntensity)

    @field_validator("smoke_color", mode="before")
    @classmethod
    def _normalize_smoke_color(cls, value: object) -> Optional[SmokeColor]:
        return _coerce_optional_enum(value, SmokeColor)

    @field_validator("reporter_location", mode="before")
    @classmethod
    def _normalize_reporter_location(
        cls, value: object
    ) -> Optional[ReporterLocation]:
        return _coerce_optional_enum(value, ReporterLocation)

    @field_validator("event_ended", mode="before")
    @classmethod
    def _normalize_event_state(cls, value: object) -> Optional[EventState]:
        return _coerce_optional_enum(value, EventState)

    @field_validator("symptoms", mode="before")
    @classmethod
    def _normalize_symptom_list(cls, value: object) -> list[SymptomValue]:
        if _is_null_like(value):
            return []
        if isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]
        normalized: list[SymptomValue] = []
        for item in items:
            if isinstance(item, (list, tuple, set)):
                for nested in item:
                    if _is_null_like(nested):
                        continue
                    normalized.append(
                        nested if isinstance(nested, SymptomValue) else SymptomValue(str(nested))
                    )
                continue
            if _is_null_like(item):
                continue
            normalized.append(item if isinstance(item, SymptomValue) else SymptomValue(str(item)))
        return normalized

    @field_validator(
        "location",
        "comments",
        "event_date",
        "event_time",
        "reporter_name",
        "home_address",
        mode="before",
    )
    @classmethod
    def _normalize_nullable_text(cls, value: object) -> Optional[str]:
        return _null_to_none(value)

    @model_validator(mode="before")
    @classmethod
    def _merge_symptom_slots(cls, data: Any) -> Any:
        """Accept legacy payloads with symptom1/2/3 and convert to a list."""
        if not isinstance(data, dict):
            return data
        payload: Dict[str, Any] = dict(data)
        aggregated: list[Any] = []
        existing = payload.get("symptoms")
        if isinstance(existing, list):
            aggregated.extend(existing)
        elif existing not in (None, "", NULL_TOKENS):
            aggregated.append(existing)
        for field in SYMPTOM_FIELD_NAMES:
            value = payload.pop(field, None)
            if value not in (None, "", NULL_TOKENS):
                aggregated.append(value)
        payload["symptoms"] = aggregated
        return payload

    @model_validator(mode="after")
    def _enforce_dependencies(self) -> "HazardReport":
        """Validate field requirements across the schema."""
        if self.smell_type and self.report_type != ReportType.ODOR:
            raise ValueError("smellType requires reportType=מפגע ריח.")
        if self.burning_material and (
            self.report_type != ReportType.ODOR or self.smell_type != SmellType.FIRE
        ):
            raise ValueError(
                "burningMaterial requires reportType=מפגע ריח and smellType=שריפה."
            )
        if self.smell_intensity and self.report_type != ReportType.ODOR:
            raise ValueError("smellIntensity requires reportType=מפגע ריח.")
        if self.symptoms and self.report_type != ReportType.ODOR:
            raise ValueError("Symptoms are only valid when reportType=מפגע ריח.")
        if (
            self.reporter_location
            and self.report_type not in {ReportType.WASTE, ReportType.SMOKE}
        ):
            raise ValueError(
                "reporterLocation requires reportType in {מפגע פסולת, מפגע עשן}."
            )
        if (self.event_date or self.event_time) and self.event_ended != EventState.ENDED:
            raise ValueError("eventDate/eventTime require eventEnded=1.")
        return self
