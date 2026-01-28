from typing import Dict, Mapping, Sequence

# Generic literal tokens reused across schema specs.
NULL_LITERAL: str = "null"
STRING_LITERAL: str = "string"
DATE_FORMAT_LITERAL: str = "YYYY-MM-DD"
TIME_FORMAT_LITERAL: str = "HH:MM:SS"


# Enumerations describing user-facing categorical choices.
REPORT_TYPES: Sequence[str] = ("מפגע ריח", "מפגע פסולת", "מפגע עשן")
SMELL_TYPES: Sequence[str] = ("שריפה", "קמין עצים", "כימי", "שפכים", "חקלאי", "אחר")
BURNING_MATERIALS: Sequence[str] = ("פלסטיק/ניילון", "צמיגים", "כימיקלים", "גזם", "לא ידוע")
SMELL_INTENSITY_VALUES: Sequence[str] = ("1", "2", "3", "4", "5", "6")
SYMPTOM_ALLOWED_VALUES: Sequence[str] = (
    "קשיי נשימה",
    "צריבה בעיניים",
    "גירוי בעור",
    "סחרחורת",
    "כאב ראש",
    "בחילה",
    "כאב או גירוי בגרון",
    "אחר",
)
SMOKE_COLORS: Sequence[str] = ("שחור", "אפור", "לבן", "אין עשן")
REPORTER_LOCATIONS: Sequence[str] = ("במקום המפגע", "רחוק מהמפגע")
EVENT_STATE_VALUES: Sequence[str] = ("0", "1")

# Field order consumed by training/prediction loops.
FIELD_REPORT_TYPE = "reportType"
FIELD_SMELL_TYPE = "smellType"
FIELD_BURNING_MATERIAL = "burningMaterial"
FIELD_SMELL_INTENSITY = "smellIntensity"
FIELD_SYMPTOMS = "symptoms"
FIELD_LOCATION = "location"
FIELD_SMOKE_COLOR = "smokeColor"
FIELD_REPORTER_LOCATION = "reporterLocation"
FIELD_EVENT_ENDED = "eventEnded"
FIELD_EVENT_DATE = "eventDate"
FIELD_EVENT_TIME = "eventTime"
FIELD_REPORTER_NAME = "reporterName"
FIELD_HOME_ADDRESS = "homeAddress"
FIELD_COMMENTS = "comments"

FIELDS: Sequence[str] = (
    FIELD_REPORT_TYPE,
    FIELD_SMELL_TYPE,
    FIELD_BURNING_MATERIAL,
    FIELD_SMELL_INTENSITY,
    FIELD_SYMPTOMS,
    FIELD_LOCATION,
    FIELD_SMOKE_COLOR,
    FIELD_REPORTER_LOCATION,
    FIELD_EVENT_ENDED,
    FIELD_EVENT_DATE,
    FIELD_EVENT_TIME,
    FIELD_REPORTER_NAME,
    FIELD_HOME_ADDRESS,
    FIELD_COMMENTS,
)


# Declarative schema definitions for each field.
FIELD_SPECS: Dict[str, Dict[str, object]] = {
    "reportType": {
        "description": "Overall hazard class selected by the reporter.",
        "allowed_values": [*REPORT_TYPES, NULL_LITERAL],
    },
    "smellType": {
        "description": (
            "Smell family for odor nuisances; constrain to enums unless unknown."
        ),
        "allowed_values": [*SMELL_TYPES, NULL_LITERAL],
        "requires": "reportType=מפגע ריח",
    },
    "burningMaterial": {
        "description": "Dominant material burning when smellType=='שריפה'.",
        "allowed_values": [*BURNING_MATERIALS, NULL_LITERAL],
        "requires": "reportType=מפגע ריח & smellType=שריפה",
    },
    "smellIntensity": {
        "description": (
            "Integer scale 1..6 describing odor strength (1=חלש מאוד, 2=חלש, 3=בינוני, "
            "4=חזק, 5=חזק מאוד, 6=בלתי נסבל); null when unspecified."
        ),
        "allowed_values": [*SMELL_INTENSITY_VALUES, NULL_LITERAL],
        "requires": "reportType=מפגע ריח",
    },
    "symptoms": {
        "description": "Ordered list of physiological symptoms mentioned in the report.",
        "allowed_values": [*SYMPTOM_ALLOWED_VALUES, NULL_LITERAL],
        "requires": "reportType=מפגע ריח",
    },
    "location": {
        "description": (
            "Literal transcription of the hazard location; transcribe verbatim when precise "
            "and never infer or geocode missing context."
        ),
        "allowed_values": [STRING_LITERAL, NULL_LITERAL],
    },
    "smokeColor": {
        "description": "Visual smoke color for waste/smoke hazards.",
        "allowed_values": [*SMOKE_COLORS, NULL_LITERAL],
        "requires": "reportType in {מפגע פסולת, מפגע עשן}",
    },
    "reporterLocation": {
        "description": "Reporter position relative to the nuisance source.",
        "allowed_values": [*REPORTER_LOCATIONS, NULL_LITERAL],
        "requires": "reportType in {מפגע פסולת, מפגע עשן}",
    },
    "eventEnded": {
        "description": "0 ongoing, 1 ended, null when unspecified.",
        "allowed_values": [*EVENT_STATE_VALUES, NULL_LITERAL],
    },
    "eventDate": {
        "description": "Calendar date formatted as YYYY-MM-DD when the eventEnded flag is 1.",
        "allowed_values": [DATE_FORMAT_LITERAL, NULL_LITERAL],
        "requires": "eventEnded=1",
    },
    "eventTime": {
        "description": "Local time formatted as HH:MM:SS when the eventEnded flag is 1.",
        "allowed_values": [TIME_FORMAT_LITERAL, NULL_LITERAL],
        "requires": "eventEnded=1",
    },
    "reporterName": {
        "description": "Reporter full name as 'First Last'; null when absent or redacted.",
        "allowed_values": [STRING_LITERAL, NULL_LITERAL],
    },
    "homeAddress": {
        "description": "Full home address (street, number, city); null when missing.",
        "allowed_values": [STRING_LITERAL, NULL_LITERAL],
    },
    "comments": {
        "description": "Free-form additional comments; null when absent.",
        "allowed_values": [STRING_LITERAL, NULL_LITERAL],
    },
}


def get_allowed_values(field: str) -> Sequence[object]:
    """Return allowed values for the requested field.

    Args:
        field: Schema field name (must exist in ``FIELD_SPECS``).

    Returns:
        Sequence containing the allowed literal values for the field.
    """
    spec: Mapping[str, object] = FIELD_SPECS[field]
    return spec.get("allowed_values", [])
