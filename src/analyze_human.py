"""Human side of proposal 2: w/na Ukrainie in the Sejm, Nov 2019 to Apr 2025 (Polish Parliamentary Corpus)."""
import json, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

df = pd.read_json(sys.argv[1] if len(sys.argv) > 1 else "data/sejm_matches.jsonl", lines=True)
df["date"] = pd.to_datetime(df["date"])
INV, GOV = pd.Timestamp("2022-02-24"), pd.Timestamp("2023-11-13")
df["period"] = np.select([df.date < INV, df.date < GOV], ["pre-invasion", "post, PiS gov"], "post, KO-led gov")
df["w"] = (df.prep == "w").astype(int)
out = {}

ctrl = df.groupby("country").agg(n=("w", "size"), share_w=("w", "mean")).round(3)
ctrl_post = df[df.date >= INV].groupby("country").agg(n=("w", "size"), share_w=("w", "mean")).round(3)
print("All countries, whole period:\n", ctrl, "\n\nAfter the invasion:\n", ctrl_post)
out["controls_post"] = ctrl_post.reset_index().to_dict("records")

ua = df[df.country == "Ukraina"].copy()
by_period = ua.groupby("period").agg(n=("w", "size"), share_w=("w", "mean")).round(3)
print("\nUkraine by period:\n", by_period)
out["ua_by_period"] = by_period.reset_index().to_dict("records")

post = ua[ua.date >= INV].copy()
post["party2"] = post.party.fillna("no club")
tab = post.pivot_table(index="party2", columns="period", values="w", aggfunc=["size", "mean"]).round(3)
print("\nUkraine after the invasion, by party and government:\n", tab)
out["ua_party_period"] = {f"{p}|{per}": dict(n=int(tab[("size", per)][p]), share_w=float(tab[("mean", per)][p]))
                          for p in tab.index for per in ("post, PiS gov", "post, KO-led gov") if not pd.isna(tab[("size", per)][p])}

# previous-speaker effect: logistic regression with party, government period, committee vs plenary, and time trend
m = post[post.prev_other_form.notna()].copy()
m["prev_w"] = (m.prev_other_form == "w").astype(int)
m["months"] = (m.date - INV).dt.days / 30.4
m["committee"] = (m.kind == "committee").astype(int)
mod = smf.logit("w ~ prev_w + C(party2) + C(period) + committee + months", data=m).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(m.speaker)[0]})
or_ = np.exp(mod.params["prev_w"]); ci = np.exp(mod.conf_int().loc["prev_w"])
print(f"\nPrevious speaker said 'w' -> odds ratio {or_:.2f} [{ci[0]:.2f}, {ci[1]:.2f}], n={len(m)}, speakers={m.speaker.nunique()}")
raw = m.groupby("prev_other_form").w.agg(["size", "mean"]).round(3)
print(raw)
out["prev_speaker"] = dict(odds_ratio=round(or_, 2), ci=[round(ci[0], 2), round(ci[1], 2)], n=len(m),
                           raw={k: dict(n=int(v["size"]), share_w=float(v["mean"])) for k, v in raw.iterrows()})

# individuals: consistency and switching
sp = post.groupby("speaker").w.agg(["size", "sum"])
sp5 = sp[sp["size"] >= 5]
mixed = int(((sp5["sum"] > 0) & (sp5["sum"] < sp5["size"])).sum())
print(f"\nSpeakers with >=5 uses: {len(sp5)}; always na {int((sp5['sum']==0).sum())}, always w {int((sp5['sum']==sp5['size']).sum())}, mixed {mixed}")
out["speakers5"] = dict(n=len(sp5), always_na=int((sp5["sum"] == 0).sum()), always_w=int((sp5["sum"] == sp5["size"]).sum()), mixed=mixed)

# half-year series by party for the figure
ua["half"] = ua.date.dt.year.astype(str) + np.where(ua.date.dt.month <= 6, "-H1", "-H2")
ser = ua[ua.party.notna()].groupby(["half", "party"]).w.agg(["size", "mean"]).reset_index()
ser.to_csv("outputs/human_halfyear_party.csv", index=False)
tot = ua.groupby("half").w.agg(["size", "mean"]).reset_index(); tot.to_csv("outputs/human_halfyear_all.csv", index=False)
json.dump(out, open("outputs/human_results.json", "w"), indent=1, ensure_ascii=False, default=str)
print("\nsaved outputs/human_results.json and the half-year series")
