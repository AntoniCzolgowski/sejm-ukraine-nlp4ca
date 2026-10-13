"""Extract every 'w/na <country, locative>' use from the Polish Parliamentary Corpus (PPC) TEI dumps.

Input: PPC term zips (e.g. 2019-2023-tei.zip, 2023-2027-tei.zip) and Sejm API MP lists (MP_term9.json, MP_term10.json).
Output: matches.jsonl (one row per match, with speaker, party, date, context) and turns.jsonl.gz (all speaker turns
of documents that contain at least one Ukraine match, kept for the replay experiment).
"""
import zipfile, json, re, sys, gzip, unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict

TEI = "{http://www.tei-c.org/ns/1.0}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"

# locative forms of country names that traditionally take "na" in Polish
COUNTRIES = {
    "Ukraina": "Ukrainie", "Litwa": "Litwie", "Białoruś": "Białorusi", "Słowacja": "Słowacji",
    "Węgry": "Węgrzech", "Łotwa": "Łotwie",
}
PAT = re.compile(r"(?<![\w-])(w|we|na)\s+(" + "|".join(COUNTRIES.values()) + r")(?![\w-])", re.I)
LOC2COUNTRY = {v.lower(): k for k, v in COUNTRIES.items()}

def norm(s):
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()

def load_mps(paths):
    """name -> list of (term, id, club, birthDate, district, education, profession)"""
    idx = defaultdict(list)
    for term, p in paths:
        for m in json.load(open(p)):
            key = norm(m["firstLastName"]).lower()
            idx[key].append(dict(term=term, mp_id=m["id"], club=m.get("club"), birthDate=m.get("birthDate"),
                                 district=m.get("districtName"), education=m.get("educationLevel"),
                                 profession=m.get("profession"), name=m["firstLastName"]))
    return idx

ROLE_PREFIXES = ["Poseł Sprawozdawca", "Posłanka Sprawozdawczyni", "Poseł", "Posłanka", "Marszałek Sejmu", "Marszałek",
                 "Wicemarszałek", "Prezes Rady Ministrów", "Wiceprezes Rady Ministrów", "Minister", "Sekretarz Stanu",
                 "Podsekretarz Stanu", "Prezydent Rzeczypospolitej Polskiej", "Senator", "Przewodniczący", "Przewodnicząca"]

def role_of(persname):
    for r in ROLE_PREFIXES:
        if persname.startswith(r):
            return r
    return "other"

def match_mp(persname, idx, term):
    """Try to find an MP whose 'First Last' is a suffix of the persName (titles come first)."""
    toks = norm(persname).split(" ")
    for k in (3, 2, 4):  # most names are 2 tokens; some have 3 (double first or last names)
        if len(toks) >= k:
            cand = " ".join(toks[-k:]).lower()
            hits = [h for h in idx.get(cand, []) if h["term"] == term] or idx.get(cand, [])
            if hits:
                return hits[0]
    return None

PARTY_RULES = [("PrawoiSprawiedliwo", "PiS"), ("KoalicjaObywatelska", "KO"), ("Lewic", "Lewica"),
               ("KoalicjaPolskaPSL", "PSL"), ("PolskieStronnictwoLudowe", "PSL"), ("Polska2050", "Polska2050"),
               ("Konfederacja", "Konfederacja"), ("Kukiz", "Kukiz15")]

def party_of(ref):
    if not ref:
        return None
    for key, lab in PARTY_RULES:
        if key in ref:
            return lab
    return "other:" + ref

def parse_doc(z, base):
    hx = z.read(base + "header.xml")
    header = ET.fromstring(hx)
    meta = {"id": base.rstrip("/").split("/")[-1]}
    bibl = header.find(f".//{TEI}sourceDesc/{TEI}bibl")
    for note in bibl.findall(TEI + "note"):
        meta[note.get("type")] = (note.text or "").strip()
    meta["date"] = (bibl.find(TEI + "date").text or "").strip()
    meta["title"] = norm("".join(bibl.find(TEI + "title").itertext()))
    persons = {}
    for p in header.iter(TEI + "person"):
        pn = p.find(TEI + "persName")
        if pn is None:
            continue
        role = pn.find(TEI + "roleName"); fn = pn.find(TEI + "forename"); sn = pn.find(TEI + "surname")
        aff = p.find(TEI + "affiliation"); sex = p.find(TEI + "sex"); wiki = p.find(TEI + "idno")
        persons[p.get(XMLID)] = dict(
            role=norm(role.text) if role is not None and role.text else "",
            name=norm(" ".join(x.text for x in (fn, sn) if x is not None and x.text)),
            sex=sex.get("value") if sex is not None else None,
            wiki=wiki.text if wiki is not None else None,
            party=party_of(aff.get("ref").lstrip("#") if aff is not None else None))
    body = ET.fromstring(z.read(base + "text_structure.xml"))
    turns = []
    for u in body.iter(TEI + "u"):
        who = (u.get("who") or "").lstrip("#")
        txt = norm("".join(u.itertext()))
        if not txt or who == "komentarz":
            continue
        if turns and turns[-1]["who"] == who:
            turns[-1]["text"] += "\n" + txt
        else:
            pp = persons.get(who, {})
            turns.append(dict(who=who, name=pp.get("name") or who, role=pp.get("role", ""), party=pp.get("party"),
                              sex=pp.get("sex"), wiki=pp.get("wiki"), text=txt))
    return meta, turns

def main(zips, mp_paths, out_prefix):
    idx = load_mps(mp_paths)
    nmatch = 0
    with open(out_prefix + "matches.jsonl", "w") as fm, gzip.open(out_prefix + "turns.jsonl.gz", "wt") as ft:
        for zp, term in zips:
            z = zipfile.ZipFile(zp)
            bases = sorted({n[:-len("text_structure.xml")] for n in z.namelist() if n.endswith("text_structure.xml")})
            for i, base in enumerate(bases):
                if "/sejm/" not in base:   # Sejm only (plenary and committees)
                    continue
                try:
                    meta, turns = parse_doc(z, base)
                except Exception as e:
                    print("SKIP", base, e, file=sys.stderr); continue
                kind = "plenary" if "/posiedzenia/" in base else "committee"
                doc_has_ua = False
                prev_form, last_form_by = None, None
                for ti, t in enumerate(turns):
                    for m in PAT.finditer(t["text"]):
                        prep = m.group(1).lower(); prep = "w" if prep in ("w", "we") else "na"
                        country = LOC2COUNTRY[m.group(2).lower()]
                        mp = match_mp(t["name"], idx, term) if t["name"] else None
                        s0 = max(0, m.start() - 300); s1 = min(len(t["text"]), m.end() + 200)
                        row = dict(doc=meta["id"], date=meta["date"], term=term, kind=kind, doc_type=meta.get("type"),
                                   title=meta.get("title"), turn=ti, who=t["who"], speaker=t["name"], role=t["role"],
                                   party=t["party"], sex=t["sex"], mp_id=mp and mp["mp_id"],
                                   country=country, prep=prep, offset=m.start(), context=t["text"][s0:s1],
                                   prev_other_form=prev_form if (prev_form and last_form_by != t["who"]) else None,
                                   preceding_word=(t["text"][:m.start()].split() or [""])[-1])
                        fm.write(json.dumps(row, ensure_ascii=False) + "\n"); nmatch += 1
                        if country == "Ukraina":
                            doc_has_ua = True
                    forms = [("w" if mm.group(1).lower() in ("w", "we") else "na") for mm in PAT.finditer(t["text"])
                             if LOC2COUNTRY[mm.group(2).lower()] == "Ukraina"]
                    if forms:
                        prev_form, last_form_by = forms[-1], t["who"]
                if doc_has_ua:
                    ft.write(json.dumps(dict(doc=meta["id"], date=meta["date"], term=term, kind=kind, title=meta.get("title"),
                                             session=meta.get("sessionNo"), day=meta.get("dayNo"), turns=turns), ensure_ascii=False) + "\n")
                if i % 500 == 0:
                    print(zp, i, len(bases), "matches so far", nmatch, flush=True)
    print("DONE matches", nmatch)

if __name__ == "__main__":
    # usage: extract_sejm.py out_prefix zip:term [zip:term ...] -- MP_term9.json:9 MP_term10.json:10
    args = sys.argv[1:]
    out_prefix = args[0]
    sep = args.index("--")
    zips = [(a.split(":")[0], int(a.split(":")[1])) for a in args[1:sep]]
    mps = [(int(a.split(":")[1]), a.split(":")[0]) for a in args[sep + 1:]]
    main(zips, mps, out_prefix)
