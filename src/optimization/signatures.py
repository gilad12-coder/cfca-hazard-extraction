from typing import Literal, Optional, List
import dspy

from .constants import (
    FIELD_BURNING_MATERIAL,
    FIELD_COMMENTS,
    FIELD_EVENT_DATE,
    FIELD_EVENT_ENDED,
    FIELD_EVENT_TIME,
    FIELD_HOME_ADDRESS,
    FIELD_LOCATION,
    FIELD_REPORTER_LOCATION,
    FIELD_REPORTER_NAME,
    FIELD_REPORT_TYPE,
    FIELD_SMELL_INTENSITY,
    FIELD_SMELL_TYPE,
    FIELD_SMOKE_COLOR,
    FIELD_SYMPTOMS,
)


# --- 1. CORE CLASSIFICATION SIGNATURES ---

class ReportTypeSignature(dspy.Signature):
    """
    Classify the report into exactly one category based on the primary hazard described.

    CLASSIFICATION LOGIC:
    1. "מפגע ריח" (Odor):
       - User reports smelling something (burning, chemicals, stench).
       - User mentions "It smells like..." or "There is a smell outside".
    2. "מפגע פסולת" (Waste):
       - User reports seeing a pile of waste/trash/debris.
       - INCLUDES: Waste piles that are burning (if the focus is the pile).
       - EXCLUDES: General complaints about "someone burning trash" without a specific location/pile context (set to null).
    3. "מפגע עשן" (Smoke):
       - User reports seeing smoke or fire, but does NOT associate it with a specific waste pile they are standing next to.
    4. "null":
       - General air pollution complaints.
       - Reports about policy (e.g., "They always burn here") without a current specific incident.
       - Reports about "burning waste" actions without a sensory description of a current hazard.
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    reportType: Literal["מפגע ריח", "מפגע פסולת", "מפגע עשן", "null"] = dspy.OutputField(
        desc="The classification category."
    )


class SmellTypeSignature(dspy.Signature):
    """
    Identify the specific source/type of the odor. Return "null" if reportType is not Odor.

    MAPPING RULES:
    - "שריפה" (Fire):
        - Includes: Smoke smell, burnt trash/garbage, burnt plastic, burnt wood/vegetation.
        - Keywords: "ריח שרוף", "ריח עשן", "ריח מדורה", "פלסטיק שרוף", "זבל שרוף".
    - "קמין עצים" (Wood Stove): Explicit mention of domestic wood ovens/heating.
    - "כימי" (Chemical):
        - Includes: Pesticides ("הדברה"), synthetic smells, paint, solvents.
    - "שפכים" (Sewage):
        - Includes: Sewage ("ביוב"), Sulfur ("גופרית"), Feces.
    - "חקלאי" (Agricultural):
        - Includes: Manure ("זבל עופות/פרות"), Fertilizer ("דשן"), Coops ("לולים/רפת").
    - "אחר" (Other):
        - Includes: Fuel ("דלק"), Cigarettes ("עישון"), Dead animals.
        - EXCLUDES: Generic "Garbage smell" (unburnt trash) -> Do not classify (return null).
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    smellType: Literal["שריפה", "קמין עצים", "כימי", "שפכים", "חקלאי", "אחר", "null"] = (
        dspy.OutputField(desc="Specific odor source.")
    )


class BurningMaterialSignature(dspy.Signature):
    """
    Identify the material being burned.
    VALIDITY: Only valid if reportType="מפגע ריח" AND smellType="שריפה". Otherwise "null".

    OPTIONS:
    - "פלסטיק/ניילון": Synthetic smell, burnt plastic/nylon.
    - "צמיגים": Burnt rubber/tires.
    - "כימיקלים": Burning electronic waste, burning chemicals.
    - "גזם": Burning wood, leaves, vegetation, "Bonfire" (מדורה), "Lag Ba'Omer".
    - "לא ידוע": User mentions "burning smell" or "fire" but does not specify the material.
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    burningMaterial: Literal[
        "פלסטיק/ניילון", "צמיגים", "כימיקלים", "גזם", "לא ידוע", "null"
    ] = dspy.OutputField(desc="Material on fire.")


class SmellIntensitySignature(dspy.Signature):
    """
    Extract the intensity of the smell as an integer 1-6 based on Hebrew descriptors.

    MAPPING:
    - 1: "חלש מאוד", "בקושי מורגש" (Barely felt).
    - 2: "חלש", "ריח קל" (Weak/Light).
    - 3: "בינוני" (Moderate).
    - 4: "חזק", "כבד", "חריף" (Strong/Heavy).
    - 5: "חזק מאוד", "כבד מאוד", "חריף ביותר" (Very strong).
    - 6: "בלתי נסבל", "קיצוני", "מחניק", "אי אפשר לנשום", "שורף בעיניים" (Unbearable/Extreme).
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    smellIntensity: Optional[int] = dspy.OutputField(desc="Integer 1-6 representing intensity.")


# --- 2. SYMPTOM EXTRACTION ---

class SymptomsSignature(dspy.Signature):
    """
    Extract all physiological symptoms mentioned and normalize to the list below.

    NORMALIZATION MAP:
    - "קשיי נשימה": Coughing, shortness of breath, asthma attack, choking, heavy breathing.
    - "צריבה בעיניים": Burning eyes, stinging eyes, tears.
    - "גירוי בעור": Itching, rash, stinging skin.
    - "סחרחורת": Dizziness, feeling faint.
    - "כאב ראש": Headache, migraine.
    - "בחילה": Nausea, vomiting.
    - "כאב או גירוי בגרון": Sore throat, itchy throat, stinging throat.
    - "אחר": Any other physiological pain/reaction.
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    symptoms: List[str] = dspy.OutputField(desc="List of normalized symptom strings.")


# --- 3. LOCATION & CONTEXT SIGNATURES ---

class LocationSignature(dspy.Signature):
    """
    Extract the location of the HAZARD.
    - Include: City, Neighborhood, Street name.
    - Valid: "Street name without number" is sufficient.
    - Exclude: Reporter's home address *unless* they explicitly state the hazard is "at my house" or "in my yard".
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    location: str = dspy.OutputField(desc="Location string or null.")


class SmokeColorSignature(dspy.Signature):
    """
    Extract the color of the smoke if visible.
    VALIDITY: Only relevant for "מפגע פסולת" (Waste) or "מפגע עשן" (Smoke).

    OPTIONS:
    - "שחור" (Black)
    - "אפור" (Gray)
    - "לבן" (White)
    - "אין עשן" (No smoke visible)
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    smokeColor: Literal["שחור", "אפור", "לבן", "אין עשן", "null"] = dspy.OutputField(
        desc="Smoke color."
    )


class ReporterLocationSignature(dspy.Signature):
    """
    Determine the reporter's proximity to the hazard source.
    - "במקום המפגע" (At the scene): Reporter sees a specific pile of waste (burning or not) close up.
    - "רחוק מהמפגע" (Far): Reporter sees smoke/fire from a distance, or smells it from their home/car.
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    reporterLocation: Literal["במקום המפגע", "רחוק מהמפגע", "null"] = dspy.OutputField(
        desc="Reporter proximity."
    )


# --- 4. METADATA / PII SIGNATURES ---

class EventEndedSignature(dspy.Signature):
    """
    Determine if the event has ended.
    - "1": Report describes a past event ("It smelled yesterday", "The fire was put out", past tense verbs).
    - "0": Report describes an ongoing event ("Smells now", "Burning right now").
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    eventEnded: Literal["0", "1", "null"] = dspy.OutputField(desc="0 for ongoing, 1 for ended.")


class EventDateSignature(dspy.Signature):
    """
    Extract the date of the event (DD/MM/YYYY) ONLY if the event has ended (eventEnded="1").
    If ongoing, return "null".
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    eventDate: str = dspy.OutputField(desc="DD/MM/YYYY string or null.")


class EventTimeSignature(dspy.Signature):
    """
    Extract the time of the event (HH:MM) ONLY if the event has ended (eventEnded="1").
    If ongoing, return "null".
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    eventTime: str = dspy.OutputField(desc="HH:MM string or null.")


class ReporterNameSignature(dspy.Signature):
    """
    Extract the First and Last name of the reporter.
    - Only extract if the user explicitly signs off or introduces themselves ("My name is X", "From Y").
    - Do not extract names of third parties.
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    reporterName: str = dspy.OutputField(desc="Full name or null.")


class HomeAddressSignature(dspy.Signature):
    """
    Extract the reporter's home address (Street + Number + City).
    - Only extract if explicitly provided as their residence ("I live at...", "Address: ...").
    """
    incident_text: str = dspy.InputField(desc="Raw Hebrew incident narrative.")
    homeAddress: str = dspy.OutputField(desc="Reporter home address or null.")


class CommentsSignature(dspy.Signature):
    """
    Extract relevant contextual details from Hebrew environmental hazard reports that do not fit into standard structured fields.

    Focus on extracting:
    1. Temporal context (e.g., "happening every night", "started 2 hours ago", "recurring").
    2. Specific source identification (e.g., "from the illegal dump", "neighbors in X village").
    3. Impact on daily life (e.g., "cannot open windows", "woke us up", "cannot breathe").
    4. Specific material descriptions if not standard (e.g., "rubber", "medical waste").

    Do not repeat the report type or standard intensity unless it adds necessary context.
    """
    incident_text: str = dspy.InputField(desc="Raw incident narrative in Hebrew.")
    comments: str = dspy.OutputField(
        desc="Concise Hebrew summary of the additional context. Return an empty string if no relevant extra info exists.")

# --- MAPPING ---

FIELD_SIGNATURES = {
    FIELD_REPORT_TYPE: ReportTypeSignature,
    FIELD_SMELL_TYPE: SmellTypeSignature,
    FIELD_BURNING_MATERIAL: BurningMaterialSignature,
    FIELD_SMELL_INTENSITY: SmellIntensitySignature,
    FIELD_SYMPTOMS: SymptomsSignature,
    FIELD_LOCATION: LocationSignature,
    FIELD_SMOKE_COLOR: SmokeColorSignature,
    FIELD_REPORTER_LOCATION: ReporterLocationSignature,
    FIELD_EVENT_ENDED: EventEndedSignature,
    FIELD_EVENT_DATE: EventDateSignature,
    FIELD_EVENT_TIME: EventTimeSignature,
    FIELD_REPORTER_NAME: ReporterNameSignature,
    FIELD_HOME_ADDRESS: HomeAddressSignature,
    FIELD_COMMENTS: CommentsSignature,
}
