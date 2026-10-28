# -*- coding: utf-8 -*-
"""Budget set — the measurement stand for prefetch_max_tokens (0.21.0), run by tools/provenance_probe.py --set budget.

What it measures, per budget: whether the labelled fact still comes back before the turn, how many other facts come
with it, whether an old version of a changed fact shows up, how many approximate tokens the block costs, how many
facts the budget left out, and how long recall takes. No network: rerank and embeddings stay off.

Synthetic data, English only. Everyone here is invented (no real people, no family data): a small flat-share and two
work projects. Slices:
  direct     — the question shares the fact's key words;
  paraphrase — mostly other words for the same thing (lexical recall may miss: that is measured, not hidden);
  namesake   — two people share a first name (Ada Quill / Ada Pell, Niko Hart / Niko Varela); `confused` labels the
               other one's fact;
  changed    — the fact was stored, then changed: the label is the fact (its current version), `stale` is the old
               text, which must not come back;
  none       — nothing in the store answers it (`targets` = []): anything recalled is noise;
  long       — the answer is a long fact full of conditions, dates and negations, the kind a cut would break.
               One of them (kestrel_brief) is longer than the 300-character line cap recall already had before 0.21.0.
Split: dev (23) to look at while choosing budget values, test (17) held out — not used to pick anything.
Labels are fact keys; the harness maps them to ids as it stores the facts.
"""

# key -> (text, entities)
FACTS = {
    "tomas_job": ("Tomas Brenner works as a bridge inspector for the city.", ["Tomas Brenner"]),
    "tomas_coffee": ("Tomas Brenner drinks his coffee black, no sugar.", ["Tomas Brenner"]),
    "tomas_swim": ("Tomas Brenner goes swimming on Thursday mornings.", ["Tomas Brenner"]),
    "tomas_knee": ("Tomas Brenner injured his left knee playing football in 2025.", ["Tomas Brenner"]),
    "ilse_bd": ("Ilse Morrow's birthday is on April 12.", ["Ilse Morrow"]),
    "ilse_allergy": ("Ilse Morrow is allergic to peanuts.", ["Ilse Morrow"]),
    "kestrel_lead": ("Ilse Morrow leads the Kestrel redesign.", ["Ilse Morrow"]),
    "kestrel_due": ("The Kestrel redesign is due at the end of January.", []),
    "ada_q_job": ("Ada Quill is the accountant who files the household taxes.", ["Ada Quill"]),
    "ada_q_hours": ("Ada Quill answers emails only on weekday mornings.", ["Ada Quill"]),
    "ada_p_job": ("Ada Pell is the physiotherapist who treats Tomas for his knee.", ["Ada Pell"]),
    "ada_p_hours": ("Ada Pell's clinic is closed on Mondays.", ["Ada Pell"]),
    "niko_h_office": ("Niko Hart shares an office with Tomas at the inspection department.", ["Niko Hart"]),
    "niko_v_landlord": ("Niko Varela is the landlord of the flat on Linden Street.", ["Niko Varela"]),
    "rosa_key": ("Rosa Lind keeps a spare key to the flat.", ["Rosa Lind"]),
    "rosa_cat": ("Rosa Lind has a cat called Pepper.", ["Rosa Lind"]),
    "felix_food": ("Felix the dog eats grain-free food twice a day.", ["Felix"]),
    "felix_vet": ("Felix sees the vet at the Oak Lane clinic.", ["Felix"]),
    "wifi": ("The home Wi-Fi network is called Larkspur.", []),
    "bins": ("The recycling bins go out on Tuesday evenings.", []),
    "car": ("The shared car is a grey hatchback with a roof rack.", []),
    "tomatoes": ("The balcony tomatoes need water every second day in summer.", []),
    "boiler": ("The boiler was serviced in March 2026.", []),
    "insurance": ("The flat insurance renews every January.", []),
    "heron_db": ("The Heron app stores its data in Postgres.", []),
    "heron_lang": ("The Heron app is written in Go.", []),
    # --- changed: stored with the old text first, then fact_store change to this one ---
    "ilse_city": ("Ilse Morrow lives in Utrecht.", ["Ilse Morrow"]),
    "rent": ("The rent for the Linden Street flat is 960 euros a month.", []),
    "felix_walk": ("Felix's evening walk is at 20:30.", ["Felix"]),
    "heron_host": ("The Heron app is hosted on two virtual servers behind a load balancer.", []),
    "tomas_dentist": ("Tomas Brenner's dentist appointment is on November 10.", ["Tomas Brenner"]),
    "niko_h_role": ("Niko Hart is a senior inspector.", ["Niko Hart"]),
    # --- long: conditions, dates and negations ---
    "travel_rules": ("The team travel budget is not approved by default: flights to the Lisbon office are reimbursed "
                     "only when the trip is booked two weeks ahead, the hotel costs under 120 euros a night, and the "
                     "team lead signs the request before 2026-11-01.", []),
    "lease_end": ("The Linden Street lease can be ended with two months' notice, but not before 2027-03-01 and only "
                  "in writing; the deposit of 1800 euros comes back within thirty days unless the landlord finds "
                  "damage beyond normal wear.", []),
    "heron_release": ("Heron releases go out on Wednesdays only, never on a Friday or a public holiday; a release "
                      "needs two approvals, the database migration runs first, and if the error rate passes two "
                      "percent within an hour the release is rolled back.", []),
    "felix_meds": ("Felix takes half a tablet of the joint supplement with breakfast, not on an empty stomach, and "
                   "the dose doubles only if the vet says so after the December check-up; the tablets are in the "
                   "top kitchen drawer.", ["Felix"]),
    "tax_papers": ("The household tax return is due on May 1; Ada Quill needs the bank statements, the childcare "
                   "receipts and the rental contract by April 1, and she will not file without the signed "
                   "authorisation form, which Tomas has to post, not email.", ["Ada Quill"]),
    "kestrel_brief": ("The Kestrel redesign brief says the new palette must not use pure black for body text, the "
                      "mobile layout ships first, the desktop layout only after user testing with at least twelve "
                      "people, any change to the checkout flow needs sign-off from both Ilse Morrow and the payments "
                      "team, and the logo stays unchanged until the board meeting on 2026-12-04.", ["Ilse Morrow"]),
}

# key -> the text it had before it changed (stored first, then changed: one fact, two versions)
CHANGED_FROM = {
    "ilse_city": "Ilse Morrow lives in Ghent.",
    "rent": "The rent for the Linden Street flat is 900 euros a month.",
    "felix_walk": "Felix's evening walk is at 19:00.",
    "heron_host": "The Heron app is hosted on a single virtual server.",
    "tomas_dentist": "Tomas Brenner's dentist appointment is on November 3.",
    "niko_h_role": "Niko Hart is a junior inspector.",
}

QUERIES = {
    # ================= dev (23) =================
    "d01": {"split": "dev", "slice": "direct", "q": "What does Tomas Brenner do for work?", "targets": ["tomas_job"]},
    "d02": {"split": "dev", "slice": "direct", "q": "When is Ilse Morrow's birthday?", "targets": ["ilse_bd"]},
    "d03": {"split": "dev", "slice": "direct", "q": "What is the Wi-Fi network called?", "targets": ["wifi"]},
    "d04": {"split": "dev", "slice": "direct", "q": "Which day do the recycling bins go out?", "targets": ["bins"]},
    "d05": {"split": "dev", "slice": "direct", "q": "How does Tomas take his coffee?", "targets": ["tomas_coffee"]},
    "p01": {"split": "dev", "slice": "paraphrase", "q": "Which clinic looks after the dog?", "targets": ["felix_vet"]},
    "p02": {"split": "dev", "slice": "paraphrase", "q": "Who does our taxes?", "targets": ["ada_q_job"],
            "confused": ["ada_p_job"]},
    "p03": {"split": "dev", "slice": "paraphrase", "q": "When does Tomas do laps in the pool?",
            "targets": ["tomas_swim"]},
    "p04": {"split": "dev", "slice": "paraphrase", "q": "How often do the tomato plants get watered?",
            "targets": ["tomatoes"]},
    "n01": {"split": "dev", "slice": "namesake", "q": "When is Ada Pell's clinic closed?", "targets": ["ada_p_hours"],
            "confused": ["ada_q_hours"]},
    "n02": {"split": "dev", "slice": "namesake", "q": "When does Ada Quill answer emails?",
            "targets": ["ada_q_hours"], "confused": ["ada_p_hours"]},
    "n03": {"split": "dev", "slice": "namesake", "q": "Who is Niko Varela?", "targets": ["niko_v_landlord"],
            "confused": ["niko_h_office", "niko_h_role"]},
    "n06": {"split": "dev", "slice": "namesake", "q": "Is Ada Quill the one who files our taxes?",
            "targets": ["ada_q_job"], "confused": ["ada_p_job"]},
    "c01": {"split": "dev", "slice": "changed", "q": "Where does Ilse Morrow live?", "targets": ["ilse_city"]},
    "c02": {"split": "dev", "slice": "changed", "q": "How much is the rent for the Linden Street flat?",
            "targets": ["rent"]},
    "c03": {"split": "dev", "slice": "changed", "q": "What time is Felix's evening walk?", "targets": ["felix_walk"]},
    "x01": {"split": "dev", "slice": "none", "q": "What is the name of Ilse's sister?", "targets": []},
    "x02": {"split": "dev", "slice": "none", "q": "Which airline do we usually fly with?", "targets": []},
    "x03": {"split": "dev", "slice": "none", "q": "What is the PIN of the bank card?", "targets": []},
    "x06": {"split": "dev", "slice": "none", "q": "Thanks, that's all for now!", "targets": []},
    "l01": {"split": "dev", "slice": "long", "q": "What are the rules for the travel budget?",
            "targets": ["travel_rules"]},
    "l02": {"split": "dev", "slice": "long", "q": "How can we end the Linden Street lease?", "targets": ["lease_end"]},
    "l03": {"split": "dev", "slice": "long", "q": "When can a Heron release go out?", "targets": ["heron_release"]},
    # ================= test (17, held out) =================
    "d06": {"split": "test", "slice": "direct", "q": "What is Ilse Morrow allergic to?", "targets": ["ilse_allergy"]},
    "d07": {"split": "test", "slice": "direct", "q": "Who keeps a spare key to the flat?", "targets": ["rosa_key"]},
    "d08": {"split": "test", "slice": "direct", "q": "What language is the Heron app written in?",
            "targets": ["heron_lang"]},
    "p05": {"split": "test", "slice": "paraphrase", "q": "When was the heating last checked?", "targets": ["boiler"]},
    "p06": {"split": "test", "slice": "paraphrase", "q": "What database does Heron use?", "targets": ["heron_db"]},
    "p07": {"split": "test", "slice": "paraphrase", "q": "When is the home insurance up for renewal?",
            "targets": ["insurance"]},
    "n04": {"split": "test", "slice": "namesake", "q": "Who does Niko Hart share an office with?",
            "targets": ["niko_h_office"], "confused": ["niko_v_landlord"]},
    "n05": {"split": "test", "slice": "namesake", "q": "What does Ada Pell treat Tomas for?",
            "targets": ["ada_p_job"], "confused": ["ada_q_job"]},
    "c04": {"split": "test", "slice": "changed", "q": "How is the Heron app hosted?", "targets": ["heron_host"]},
    "c05": {"split": "test", "slice": "changed", "q": "When is Tomas Brenner's dentist appointment?",
            "targets": ["tomas_dentist"]},
    "c06": {"split": "test", "slice": "changed", "q": "Is Niko Hart a junior or a senior inspector?",
            "targets": ["niko_h_role"], "confused": ["niko_v_landlord"]},
    "x04": {"split": "test", "slice": "none", "q": "What is the capital of Australia?", "targets": []},
    "x05": {"split": "test", "slice": "none", "q": "When does the swimming pool close on Sundays?", "targets": []},
    "x07": {"split": "test", "slice": "none", "q": "What colour is Rosa's front door?", "targets": []},
    "l04": {"split": "test", "slice": "long", "q": "How much of the joint supplement does Felix take?",
            "targets": ["felix_meds"]},
    "l05": {"split": "test", "slice": "long", "q": "What does Ada Quill need for the tax return?",
            "targets": ["tax_papers"]},
    "l06": {"split": "test", "slice": "long", "q": "What does the Kestrel redesign brief say about the logo?",
            "targets": ["kestrel_brief"]},
}

SLICES = ("direct", "paraphrase", "namesake", "changed", "none", "long")


def self_check() -> None:
    """Labels are right: every target and confused key exists, `none` is exactly [], a changed target has its old
    text, the splits are frozen at 23 dev / 17 test."""
    errs = []
    for qid, q in QUERIES.items():
        if q["slice"] not in SLICES:
            errs.append(f"{qid}: unknown slice")
        if not all(t in FACTS for t in q["targets"] + q.get("confused", [])):
            errs.append(f"{qid}: unknown key")
        if (q["slice"] == "none") != (q["targets"] == []):
            errs.append(f"{qid}: none label mismatch")
        if q["slice"] == "changed" and not all(t in CHANGED_FROM for t in q["targets"]):
            errs.append(f"{qid}: changed target without an old text")
        if set(q.get("confused", [])) & set(q["targets"]):
            errs.append(f"{qid}: confused overlaps targets")
    if not set(CHANGED_FROM) <= set(FACTS):
        errs.append("CHANGED_FROM names an unknown fact")
    split = {s: sum(1 for q in QUERIES.values() if q["split"] == s) for s in ("dev", "test")}
    if split != {"dev": 23, "test": 17} or len(QUERIES) != 40:
        errs.append(f"split counts {split}")
    if errs:
        raise SystemExit("budget set self_check FAILED:\n  " + "\n  ".join(errs))


if __name__ == "__main__":
    self_check()
    print(f"budget set OK: {len(FACTS)} facts ({len(CHANGED_FROM)} changed), {len(QUERIES)} questions")
