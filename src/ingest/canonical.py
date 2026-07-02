"""Canonical team registry — the name-harmonisation gate (spec §2.3, §7.1).

The single tueur silencieux of this project is inconsistent team naming across
sources ("Korea Republic" vs "South Korea", "USA" vs "United States"). One team
mis-resolved poisons its entire Elo history, so *nothing downstream may proceed
until every name resolves*.

Design
------
The martj42 dataset (our training base) already uses one internally-consistent
name per team; we treat those spellings as the **canonical** forms. Each canonical
name gets a stable integer ``team_id`` (assigned deterministically by sorting the
canonical names) plus a ``confederation``.

On top of that we register **aliases**: the alternative spellings used by FIFA,
Fjelstul, Transfermarkt, FBref, etc. ("Korea Republic", "USA", "IR Iran",
"Czechia", "Cabo Verde", "China PR"...). ``resolve(name)`` normalises and looks a
name up through both canonical names and aliases.

Public API
----------
``resolve(name) -> int | None``           team_id or None if unresolved
``resolve_strict(name) -> int``           raises KeyError if unresolved
``canonical_name(team_id) -> str``
``confederation(team_id) -> str``
``build_team_table() -> list[dict]``       rows for the ``teams`` DB table
``unresolved_names(names) -> list[str]``   names that do NOT resolve (the gate)
"""

from __future__ import annotations

import csv
import unicodedata
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# Confederation lookup for every canonical (martj42) name.
# FIFA members map to their real confederation; non-FIFA / regional sides
# (Sápmi, Padania, Kurdistan, Yorkshire, ...) are tagged "NON-FIFA" so the
# training pipeline can optionally filter them. This keeps confederation a
# non-null, documented field for all 336 names.
# ---------------------------------------------------------------------------

# FIFA members by confederation.
_UEFA = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
    "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
    "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
    "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
    "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo", "Latvia",
    "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco",
    "Montenegro", "Netherlands", "North Macedonia", "Northern Ireland",
    "Norway", "Poland", "Portugal", "Republic of Ireland", "Romania", "Russia",
    "San Marino", "Scotland", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
}
_CONMEBOL = {
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Paraguay", "Peru", "Uruguay", "Venezuela",
}
_CONCACAF = {
    "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
    "Belize", "Bermuda", "Bonaire", "British Virgin Islands", "Canada",
    "Cayman Islands", "Costa Rica", "Cuba", "Curaçao", "Dominica",
    "Dominican Republic", "El Salvador", "French Guiana", "Grenada",
    "Guadeloupe", "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica",
    "Martinique", "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Martin",
    "Saint Vincent and the Grenadines", "Sint Maarten", "Suriname",
    "Trinidad and Tobago", "Turks and Caicos Islands", "United States",
    "United States Virgin Islands",
}
_CAF = {
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
    "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
    "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea",
    "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
    "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
    "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
    "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
    "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone",
    "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
    "Tunisia", "Uganda", "Zambia", "Zimbabwe", "Zanzibar", "Réunion",
    "Mayotte",
}
_AFC = {
    "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
    "Cambodia", "China", "Guam", "Hong Kong", "India", "Indonesia", "Iran",
    "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon",
    "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
    "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
    "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria",
    "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan",
    "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
}
_OFC = {
    "American Samoa", "Cook Islands", "Fiji", "Kiribati", "New Caledonia",
    "New Zealand", "Niue", "Papua New Guinea", "Samoa", "Solomon Islands",
    "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
}

# Defunct / historical FIFA sides that still appear in results.csv. Mapped to
# their best-fit confederation so their history is usable for Elo continuity.
_HISTORICAL = {
    "Czechoslovakia": "UEFA",
    "Yugoslavia": "UEFA",
    "German DR": "UEFA",            # East Germany
    "Saarland": "UEFA",
    "North Vietnam": "AFC",
    "Vietnam Republic": "AFC",      # South Vietnam
    "South Yemen": "AFC",
    "Yemen DPR": "AFC",             # (North) Yemen historical
    "Manchukuo": "AFC",
    "United Koreans in Japan": "AFC",
}

_CONF_SETS = {
    "UEFA": _UEFA,
    "CONMEBOL": _CONMEBOL,
    "CONCACAF": _CONCACAF,
    "CAF": _CAF,
    "AFC": _AFC,
    "OFC": _OFC,
}

# ---------------------------------------------------------------------------
# Alias map: alternative spelling  ->  canonical (martj42) name.
# Keys are matched after normalisation (case/accents/punctuation-insensitive),
# so we only need one representative spelling per variant family.
# ---------------------------------------------------------------------------
_ALIASES = {
    # --- the classic FIFA / Fjelstul / broadcast variants -------------------
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "Korea Democratic People's Republic": "North Korea",
    "USA": "United States",
    "United States of America": "United States",
    "US": "United States",
    "IR Iran": "Iran",
    "Iran (Islamic Republic of)": "Iran",
    "Czechia": "Czech Republic",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Ivory Coast (Côte d'Ivoire)": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Republic of the Congo": "Congo",
    "Congo-Brazzaville": "Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "DR Congo (Zaire)": "DR Congo",
    "Zaire": "DR Congo",
    "Congo Kinshasa": "DR Congo",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
    "Ireland": "Republic of Ireland",
    "Eire": "Republic of Ireland",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Cape Verde Is.": "Cape Verde",
    "Swaziland": "Eswatini",
    "Curacao": "Curaçao",
    "The Gambia": "Gambia",
    "Timor Leste": "Timor-Leste",
    "East Timor": "Timor-Leste",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St. Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "St Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "UAE": "United Arab Emirates",
    "Brunei Darussalam": "Brunei",
    "Myanmar (Burma)": "Myanmar",
    "Burma": "Myanmar",
    "Netherlands Antilles": "Curaçao",   # successor side used by martj42 era
    "Great Britain": "England",          # rare Olympic entries -> England record
    "West Germany": "Germany",
    "FR Germany": "Germany",
    "East Germany": "German DR",
    "GDR": "German DR",
    "Soviet Union": "Russia",            # successor mapping for Elo continuity
    "USSR": "Russia",
    "CIS": "Russia",
    "Serbia and Montenegro": "Serbia",   # successor side
    "FR Yugoslavia": "Serbia",
    "South Vietnam": "Vietnam Republic",
    "North Yemen": "Yemen DPR",
    "Chinese Taipei (Taiwan)": "Taiwan",
    "Kirghizia": "Kyrgyzstan",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Sao Tome e Principe": "São Tomé and Príncipe",
    "Reunion": "Réunion",
    "Aland Islands": "Åland Islands",
    "Vatican": "Vatican City",
    "Hong Kong, China": "Hong Kong",
    "Macau, China": "Macau",
    "Macao": "Macau",
    "Kosovo (UNMIK)": "Kosovo",
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _normalise(name: str) -> str:
    """Fold a raw name to a comparison key: strip accents, lowercase, squeeze.

    Accent-folding lets "Curacao"/"Curaçao" and "Reunion"/"Réunion" collide, so
    we don't need a separate alias for every accent stripping variant.
    """
    if name is None:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    # unify punctuation/whitespace to single spaces
    for ch in "._/&-,":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


# ---------------------------------------------------------------------------
# Registry construction (built once, cached).
# ---------------------------------------------------------------------------
# Canonical names are seeded from the martj42 results.csv shipped in data/raw
# so that *every* name in the training base is canonical by construction (the
# gate can never be broken by a hand-list omission). This module-level path is
# resolved relative to the repo root; overridable in tests via _seed_names_from.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RESULTS_CSV = _REPO_ROOT / "data" / "raw" / "martj42_results.csv"


def _seed_names_from(csv_path: Path) -> set[str]:
    """Read the distinct home/away team names from a martj42-style results CSV."""
    names: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.add(row["home_team"])
            names.add(row["away_team"])
    return names


def _all_canonical_names() -> list[str]:
    """Every canonical name, sorted for deterministic ids.

    Union of: names observed in the martj42 base (authoritative), our curated
    confederation sets, historical sides, and known non-FIFA regional sides.
    Seeding from the data guarantees no training name is ever left without an id.
    """
    names: set[str] = set()
    if _DEFAULT_RESULTS_CSV.exists():
        names |= _seed_names_from(_DEFAULT_RESULTS_CSV)
    for s in _CONF_SETS.values():
        names |= s
    names |= set(_HISTORICAL)
    names |= _NON_FIFA
    return sorted(names)


# Regional / non-FIFA teams that appear in martj42 results.csv.
_NON_FIFA = {
    "Abkhazia", "Alderney", "Ambazonia", "Andalusia", "Arameans Suryoye",
    "Artsakh", "Asturias", "Aymara", "Barawa", "Basque Country", "Biafra",
    "Brittany", "Canary Islands", "Cascadia", "Catalonia", "Central Spain",
    "Chagos Islands", "Chameria", "Chechnya", "Cilento", "Corsica",
    "County of Nice", "Crimea", "Darfur", "Donetsk PR", "Délvidék",
    "East Turkestan", "Elba Island", "Ellan Vannin", "Falkland Islands",
    "Felvidék", "Franconia", "Frøya", "Galicia", "Gotland", "Gozo",
    "Greenland", "Guernsey", "Găgăuzia", "Hitra", "Hmong", "Iraqi Kurdistan",
    "Isle of Man", "Isle of Wight", "Jersey", "Kabylia", "Kernow",
    "Kurdistan", "Kárpátalja", "Luhansk PR", "Madrid", "Mapuche",
    "Marshall Islands", "Matabeleland", "Maule Sur", "Menorca", "Micronesia",
    "Northern Cyprus", "Northern Mariana Islands", "Occitania", "Orkney",
    "Padania", "Palau", "Panjab", "Parishes of Jersey", "Provence", "Quebec",
    "Raetia", "Rhodes", "Romani people", "Ryūkyū", "Saare County",
    "Saint Barthélemy", "Saint Helena", "Saint Pierre and Miquelon", "Sark",
    "Saugeais", "Sealand", "Seborga", "Shetland", "Silesia", "Somaliland",
    "South Ossetia", "Surrey", "Sápmi", "Székely Land", "Tamil Eelam",
    "Tibet", "Ticino", "Two Sicilies", "Wallis Islands and Futuna",
    "West Papua", "Western Armenia", "Western Australia", "Western Isles",
    "Western Sahara", "Ynys Môn", "Yorkshire", "Yoruba Nation",
}


@lru_cache(maxsize=1)
def _registry() -> dict:
    """Build and cache the id/name/confederation/alias tables."""
    canon = _all_canonical_names()
    name_to_id = {name: i for i, name in enumerate(canon)}
    id_to_name = {i: name for name, i in name_to_id.items()}

    # confederation per canonical name
    conf: dict[str, str] = {}
    for c, s in _CONF_SETS.items():
        for name in s:
            conf[name] = c
    conf.update(_HISTORICAL)
    for name in _NON_FIFA:
        conf.setdefault(name, "NON-FIFA")
    # any canonical name not classified above defaults to NON-FIFA (documented)
    for name in canon:
        conf.setdefault(name, "NON-FIFA")

    # lookup: normalised key -> team_id (canonical names + aliases)
    lookup: dict[str, int] = {}
    for name, tid in name_to_id.items():
        lookup[_normalise(name)] = tid
    for alias, target in _ALIASES.items():
        if target not in name_to_id:
            raise ValueError(
                f"Alias {alias!r} targets unknown canonical name {target!r}"
            )
        lookup[_normalise(alias)] = name_to_id[target]

    return {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "conf": conf,
        "lookup": lookup,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve(name: str) -> int | None:
    """Return the ``team_id`` for *name*, or ``None`` if it cannot be resolved."""
    return _registry()["lookup"].get(_normalise(name))


def resolve_strict(name: str) -> int:
    """Like :func:`resolve` but raise ``KeyError`` on an unresolved name."""
    tid = resolve(name)
    if tid is None:
        raise KeyError(f"Unresolved team name: {name!r}")
    return tid


def canonical_name(team_id: int) -> str:
    """Return the canonical name for a ``team_id``."""
    return _registry()["id_to_name"][team_id]


def confederation(team_id: int) -> str:
    """Return the confederation code for a ``team_id``."""
    reg = _registry()
    return reg["conf"][reg["id_to_name"][team_id]]


def build_team_table() -> list[dict]:
    """Rows for the DB ``teams`` table (elo_current left None, owned by ratings)."""
    reg = _registry()
    rows = []
    for name, tid in sorted(reg["name_to_id"].items(), key=lambda kv: kv[1]):
        rows.append(
            {
                "team_id": tid,
                "name_canonical": name,
                "confederation": reg["conf"][name],
                "elo_current": None,
            }
        )
    return rows


def unresolved_names(names) -> list[str]:
    """Return the sorted list of *names* that do NOT resolve — the hard gate.

    Downstream must not proceed while this is non-empty (spec §2.3, §7.1).
    """
    seen = set()
    out = []
    for n in names:
        if resolve(n) is None and n not in seen:
            seen.add(n)
            out.append(n)
    return sorted(out)


if __name__ == "__main__":
    # Prove the gate over the full martj42 name set.
    import csv
    from src.config import load_config

    cfg = load_config()
    path = cfg.paths.raw_dir / "martj42_results.csv"
    names: set[str] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names.add(row["home_team"])
            names.add(row["away_team"])
    unresolved = unresolved_names(names)
    print(f"canonical teams: {len(build_team_table())}")
    print(f"names in martj42: {len(names)}")
    print(f"unresolved: {len(unresolved)}")
    if unresolved:
        for n in unresolved:
            print("  UNRESOLVED:", n)
        raise SystemExit(1)
    print("GATE PASSED: 0 unresolved names")
