"""Build the agent-probe cases for proposal 2 (w/na Ukrainie in the Sejm).

Each case is one real moment where an MP had to choose "w" or "na" before "Ukrainie" (first such moment in a speech).
For every case we write the prompt pieces for five conditions:
  C0 bare     : scene only
  C1 party    : scene + party
  C2 grounded : scene + party + biography + style samples + the real speeches before this one
  C3 flipped  : C2 with every w/na Ukrainie in the preceding speeches flipped
  C4 swapped  : C2 with the party label swapped to the other camp
Pre-invasion cases (all "na") get C2 with the real date and C2 with the date moved +3 years (D1).
The assistant message is pre-filled with the MP's real words up to the choice point; src/score.py compares the full
continuations " w Ukrainie" vs " na Ukrainie" (field cont, with the transcript's spacing and capitalisation).
No MP names appear in any prompt.
"""
import gzip, json, re, random, datetime as dt, unicodedata
from collections import defaultdict

random.seed(2026)
DAYS = ["poniedziałek", "wtorek", "środa", "czwartek", "piątek", "sobota", "niedziela"]
MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września", "października", "listopada", "grudnia"]
PARTY_PL = {"PiS": "Prawo i Sprawiedliwość", "KO": "Koalicja Obywatelska", "Lewica": "Lewica", "PSL": "Polskie Stronnictwo Ludowe",
            "Polska2050": "Polska 2050", "Konfederacja": "Konfederacja", "Kukiz15": "Kukiz'15"}
SWAP = {"PiS": "KO", "KO": "PiS", "Konfederacja": "Lewica", "Lewica": "Konfederacja", "PSL": "Lewica", "Polska2050": "Konfederacja", "Kukiz15": "Lewica"}
UA = re.compile(r"(?<![\w-])(w|we|na|W|We|Na)(\s+)(Ukrainie)(?![\w-])")
LOC_ANY = re.compile(r"(?<![\w-])(w|we|na)\s+(Ukrainie|Litwie|Białorusi|Słowacji|Węgrzech|Łotwie)(?![\w-])", re.I)

def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()
def words_tail(s, n): w = s.split(); return ("… " if len(w) > n else "") + " ".join(w[-n:])
def words_mid(s, n):
    w = s.split()
    if len(w) <= n: return " ".join(w)
    a = (len(w) - n) // 2; return "… " + " ".join(w[a:a + n]) + " …"
def date_pl(d): return f"{DAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]} {d.year} r."
def roman(t): return {9: "IX", 10: "X"}[t]
def flip(text):
    return UA.sub(lambda m: {"w": "na", "we": "na", "na": "w", "W": "Na", "We": "Na", "Na": "W"}[m.group(1)] + m.group(2) + m.group(3), text)

# --- MPs (bio) keyed by "first last" per term
mps = {}
for t in (9, 10):
    for m in json.load(open(f"data/MP_term{t}.json")):
        mps[(t, norm(m["firstName"] + " " + m["lastName"]).lower())] = m
def find_mp(name, term):
    key = norm(name).lower()
    return mps.get((term, key)) or mps.get((19 - term, key))

# --- documents
docs = {}
with gzip.open("out_turns.jsonl.gz", "rt") as f:
    for line in f:
        d = json.loads(line); docs[d["doc"]] = d
# speaker -> list of (date, text) of turns without any country locative and >= 60 words, for style samples
style = defaultdict(list)
for d in docs.values():
    for t in d["turns"]:
        if t.get("party") and len(t["text"].split()) >= 60 and not LOC_ANY.search(t["text"]) and "Ukrain" not in t["text"]:
            style[t["name"]].append((d["date"], t["text"]))

def agenda_before(turns, ti):
    for j in range(ti - 1, -1, -1):
        m = re.search(r"porządku dziennego:\s*([^\n]{5,260})", turns[j]["text"])
        if m:
            return norm(m.group(1)).rstrip(".")
    return None

def scene(meta, date, agenda):
    d = dt.date.fromisoformat(date)
    if meta["kind"] == "plenary":
        s = f"Jest {date_pl(d)} Trwa posiedzenie plenarne Sejmu Rzeczypospolitej Polskiej {roman(meta['term'])} kadencji. Jesteś na sali plenarnej."
        if agenda:
            s += f" Sejm rozpatruje punkt porządku dziennego: {agenda}."
        s += " Zabierasz głos z mównicy, a Twoje słowa trafią do stenogramu."
    else:
        s = f"Jest {date_pl(d)} Trwa posiedzenie sejmowej komisji ({meta['title']}), Sejm Rzeczypospolitej Polskiej {roman(meta['term'])} kadencji. Zabierasz głos na posiedzeniu, a Twoje słowa trafią do zapisu przebiegu posiedzenia."
    return s

def identity(party, sex, role, mp, samples, with_bio=True):
    pos = "posłanką" if sex == "F" else "posłem"
    s = f"Jesteś {pos} na Sejm RP z klubu {PARTY_PL[party]}."
    if role and not role.lower().startswith("pos"):
        s += f" Pełnisz funkcję: {role.lower()}."
    if with_bio and mp:
        born = "Urodziłaś się" if sex == "F" else "Urodziłeś się"
        bits = [f"{born} w {mp['birthDate'][:4]} r. w miejscowości {mp.get('birthLocation','?')}."]
        bits.append(f"Mandat zdobyłeś w okręgu {mp['districtName']} (województwo {mp.get('voivodeship','?')})." if sex != "F"
                    else f"Mandat zdobyłaś w okręgu {mp['districtName']} (województwo {mp.get('voivodeship','?')}).")
        if mp.get("educationLevel"): bits.append(f"Wykształcenie: {mp['educationLevel']}.")
        if mp.get("profession"): bits.append(f"Zawód: {mp['profession']}.")
        s += " " + " ".join(bits)
    if with_bio and samples:
        s += "\n\nTak brzmią fragmenty Twoich wcześniejszych wystąpień:\n" + "\n".join(f"„{x}”" for x in samples)
    return s

def context_block(turns, ti, k=3, nwords=90, do_flip=False):
    prev = []
    for j in range(max(0, ti - k), ti):
        t = turns[j]
        lab = t["role"] or "Mówca"
        if t.get("party"): lab += f" ({PARTY_PL.get(t['party'], t['party'])})"
        txt = words_tail(t["text"], nwords)
        prev.append(f"{lab}: „{flip(txt) if do_flip else txt}”")
    return ("Bezpośrednio przed Tobą na sali padły te słowa:\n" + "\n".join(prev)) if prev else ""

TASK = "Teraz Twoja kolej. Wygłoś swoje wystąpienie tak, jak zrobił(a)byś to naprawdę."

rows = [json.loads(l) for l in open("out_matches.jsonl")]
# first Ukraine locative per turn, MPs with a club only
seen, cand = set(), []
for r in sorted(rows, key=lambda r: (r["doc"], r["turn"], r["offset"])):
    if r["country"] != "Ukraina" or not r["party"] or r["party"].startswith("other"):
        continue
    key = (r["doc"], r["turn"])
    if key in seen: continue
    seen.add(key); cand.append(r)

post = [r for r in cand if r["date"] >= "2022-02-24"]
pre = [r for r in cand if r["date"] < "2022-02-24"]
cells = defaultdict(list)
for r in post:
    period = "PiS-gov" if r["date"] < "2023-11-13" else "KO-gov"
    cells[(r["party"], period)].append(r)
CAP = 80
sample = []
for k, v in sorted(cells.items()):
    random.shuffle(v); sample += [(r, k[1]) for r in v[:CAP]]
print("candidate first-in-turn cases: post", len(post), "pre", len(pre))
print({f"{k[0]}|{k[1]}": len(v) for k, v in sorted(cells.items())})

def make_case(r, period, kind_tag):
    d = docs[r["doc"]]; turns = d["turns"]; ti = r["turn"]; t = turns[ti]
    meta = dict(kind=d["kind"], term=d["term"], title=d["title"])
    agenda = agenda_before(turns, ti) if d["kind"] == "plenary" else None
    mp = find_mp(t["name"], d["term"])
    smp = [x for x in style.get(t["name"], []) if x[0] < d["date"]]
    random.shuffle(smp)
    samples = [words_mid(x[1], 55) for x in smp[:2]]
    prefix = words_tail(t["text"][:r["offset"]].rstrip(), 150)
    # the two continuations to score, exactly as they would sit in the transcript after the prefix:
    # a space unless the form opens a quote („w Ukrainie”), and a capital after a full stop
    before, said = t["text"][:r["offset"]], UA.match(t["text"], r["offset"])
    assert said, (r["doc"], ti)
    sep = " " if before[-1:].isspace() else ""
    cap = said.group(1)[0].isupper()
    cont = {"w": sep + ("W" if cap else "w") + " Ukrainie", "na": sep + ("Na" if cap else "na") + " Ukrainie"}
    ctx = context_block(turns, ti)
    base = dict(case_id=f"{r['doc']}#{ti}", doc=r["doc"], turn=ti, date=d["date"], term=d["term"], kind=d["kind"],
                party=r["party"], period=period, set=kind_tag, sex=t.get("sex"), role=t["role"], real=r["prep"],
                prev_other_form=r["prev_other_form"], has_bio=bool(mp), n_style=len(samples),
                agenda_form=(UA.search(agenda).group(1).lower() if agenda and UA.search(agenda) else None),
                context_forms=[m.group(1).lower() for m in UA.finditer(ctx)], prefix=prefix, cont=cont,
                quoted=bool(before) and before[-1] in "„\"«")   # the MP is quoting the form (talking about the word itself)
    sc = scene(meta, d["date"], agenda)
    sc_flip = scene(meta, d["date"], flip(agenda) if agenda else None)
    idn = identity(r["party"], t.get("sex"), t["role"], mp, samples)
    conds = {}
    if kind_tag == "post":
        conds["C0_bare"] = sc + "\n\n" + TASK
        conds["C1_party"] = sc + "\n\n" + identity(r["party"], t.get("sex"), "", None, [], with_bio=False) + "\n\n" + TASK
        conds["C2_grounded"] = "\n\n".join(x for x in (sc, idn, ctx, TASK) if x)
        conds["C3_flipped"] = "\n\n".join(x for x in (sc_flip, idn, context_block(turns, ti, do_flip=True), TASK) if x)
        conds["C4_swapped"] = "\n\n".join(x for x in (sc, identity(SWAP[r["party"]], t.get("sex"), t["role"], mp, samples), ctx, TASK) if x)
        base["swap_party"] = SWAP[r["party"]]
    else:
        d_shift = dt.date.fromisoformat(d["date"]); d_shift = d_shift.replace(year=d_shift.year + 3)
        conds["C2_grounded"] = "\n\n".join(x for x in (sc, idn, ctx, TASK) if x)
        conds["D1_date_plus3y"] = "\n\n".join(x for x in (scene(meta, d_shift.isoformat(), agenda), idn, ctx, TASK) if x)
    base["prompts"] = conds
    return base

cases = [make_case(r, p, "post") for r, p in sample] + [make_case(r, "pre-invasion", "pre") for r in pre]
with open("cases.jsonl", "w") as f:
    for c in cases: f.write(json.dumps(c, ensure_ascii=False) + "\n")
n_prompts = sum(len(c["prompts"]) for c in cases)
print("cases", len(cases), "prompts", n_prompts, "with bio", sum(c["has_bio"] for c in cases), "with style", sum(c["n_style"] > 0 for c in cases))
lens = sorted(len(p) for c in cases for p in c["prompts"].values())
print("prompt chars median", lens[len(lens)//2], "p95", lens[int(len(lens)*.95)], "max", lens[-1])
