"""Interactive demo: estimate the price of one house with the saved pipeline.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import house_features  # noqa: F401  (needed so joblib can unpickle the pipeline)

ARTIFACT_PATH = Path(__file__).parent / "models" / "house_price_pipeline.joblib"
SQFT_TO_M2 = 0.092903

NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT & Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NPkVill": "Northpark Villa", "NWAmes": "Northwest Ames",
    "NoRidge": "Northridge", "NridgHt": "Northridge Heights", "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State Univ.", "Sawyer": "Sawyer", "SawyerW": "Sawyer West",
    "Somerst": "Somerset", "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
}
QUALITY = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical/Average", "Fa": "Fair", "Po": "Poor"}
ZONING = {"RL": "Residensial kepadatan rendah", "RM": "Residensial kepadatan sedang",
          "RH": "Residensial kepadatan tinggi", "FV": "Floating Village (residensial)",
          "C (all)": "Komersial"}
BLDG = {"1Fam": "Rumah tunggal", "2fmCon": "Konversi 2 keluarga", "Duplex": "Duplex",
        "TwnhsE": "Townhouse (ujung)", "Twnhs": "Townhouse (tengah)"}


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)


def usd(v: float) -> str:
    return f"${v:,.0f}"


st.set_page_config(page_title="Estimasi Harga Rumah", page_icon="🏠", layout="wide")
artifact = load_artifact()
pipe, meta = artifact["pipeline"], artifact["metadata"]
ds, ranges, tpl = meta["dataset"], meta["feature_ranges"], meta["template"]
cv = meta["metrics_cv"][meta["model_name"]]
y0, y1 = ds["sale_years"]

st.title("🏠 Estimasi Harga Rumah — Ames, Iowa")
st.caption(f"Model: **{meta['model_name']}** · R² CV {cv['R² (CV)']:.3f} · "
           f"RMSE CV {usd(cv['RMSE USD (CV)'])}")

# ---------- Dataset scope ----------
with st.sidebar:
    st.header("Cakupan dataset")
    st.markdown(f"""
- **Mata uang:** {meta['currency']} (dolar Amerika Serikat)
- **Dasar harga:** {meta['price_basis'].lower()}
- **Lokasi:** {ds['location']}
- **Periode penjualan:** {y0}–{y1}
- **Tahun dibangun:** {ds['year_built'][0]}–{ds['year_built'][1]}
- **Jumlah rumah latih:** {ds['n_train_rows']:,}
- **Rentang harga latih:** {usd(ds['price_min'])} – {usd(ds['price_max'])}
- **Median harga latih:** {usd(ds['price_median'])}
- **Satuan luas:** {ds['area_unit']}
""")
    st.info("Estimasi hanya berlaku untuk pasar perumahan Ames pada 2006–2010. "
            "Mengonversi angka ini ke rupiah **tidak** menghasilkan harga rumah di Indonesia.")
    st.caption(f"Sumber data: {ds['name']}")


def num(label, col, step=1, unit=None, **kw):
    r = ranges[col]
    default = kw.pop("value", tpl[col])
    help_txt = f"Rentang data latih: {r['min']:,.0f} – {r['max']:,.0f}"
    if unit:
        help_txt += f" {unit}"
    if "help" in kw:
        help_txt = kw.pop("help") + " " + help_txt
    return st.number_input(label, min_value=0, max_value=int(r["max"] * 2), value=int(default),
                           step=step, help=help_txt, **kw)


# ---------- Inputs ----------
st.subheader("Karakteristik rumah")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Lokasi & tipe**")
    nb_codes = meta["categories"]["Neighborhood"]
    neighborhood = st.selectbox("Neighborhood", nb_codes, index=nb_codes.index("CollgCr"),
                                format_func=lambda c: f"{NEIGHBORHOODS.get(c, c)} ({c})")
    zoning = st.selectbox("Zonasi", meta["categories"]["MSZoning"],
                          index=meta["categories"]["MSZoning"].index(tpl["MSZoning"]),
                          format_func=lambda c: ZONING.get(c, c))
    bldg = st.selectbox("Tipe bangunan", meta["categories"]["BldgType"],
                        index=meta["categories"]["BldgType"].index(tpl["BldgType"]),
                        format_func=lambda c: BLDG.get(c, c))
    lot_area = num("Luas tanah (sq ft)", "LotArea", step=500, unit="sq ft")
    st.caption(f"≈ {lot_area * SQFT_TO_M2:,.0f} m²")
    yr_sold = st.selectbox("Tahun terjual", list(range(y0, y1 + 1)), index=y1 - y0)

with c2:
    st.markdown("**Bangunan**")
    first = num("Luas lantai 1 (sq ft)", "1stFlrSF", step=50, value=1100, unit="sq ft")
    second = num("Luas lantai 2 (sq ft)", "2ndFlrSF", step=50, value=800, unit="sq ft")
    bsmt = num("Luas basement (sq ft)", "TotalBsmtSF", step=50, value=first, unit="sq ft")
    gr_liv = first + second
    st.caption(f"Luas hunian di atas tanah: {gr_liv:,} sq ft ≈ {gr_liv * SQFT_TO_M2:,.0f} m²")
    year_built = num("Tahun dibangun", "YearBuilt", value=2005)
    year_remod = num("Tahun renovasi terakhir", "YearRemodAdd", value=max(2005, year_built),
                     help="Isi sama dengan tahun dibangun jika belum pernah direnovasi.")

with c3:
    st.markdown("**Kualitas & fasilitas**")
    overall_qual = st.slider("Kualitas keseluruhan (1–10)", 1, 10, 7)
    overall_cond = st.slider("Kondisi keseluruhan (1–10)", 1, 10, 5)
    q_opts = ["Ex", "Gd", "TA", "Fa"]
    kitchen = st.selectbox("Kualitas dapur", q_opts, index=1, format_func=QUALITY.get)
    exter = st.selectbox("Kualitas eksterior", q_opts, index=1, format_func=QUALITY.get)
    central_air = st.toggle("AC sentral", value=True)

c4, c5, c6, c7 = st.columns(4)
with c4:
    bedrooms = num("Kamar tidur", "BedroomAbvGr", value=3)
    rooms = num("Total ruangan (tanpa kamar mandi)", "TotRmsAbvGrd", value=max(7, bedrooms + 3))
with c5:
    full_bath = num("Kamar mandi lengkap", "FullBath", value=2)
    half_bath = num("Kamar mandi setengah", "HalfBath", value=1)
with c6:
    garage_cars = num("Kapasitas garasi (mobil)", "GarageCars", value=2)
    garage_area = num("Luas garasi (sq ft)", "GarageArea", step=20,
                      value=0 if garage_cars == 0 else 250 * garage_cars, unit="sq ft")
with c7:
    fireplaces = num("Perapian", "Fireplaces", value=1)

# ---------- Build one row with the same raw columns as training ----------
house = {
    **tpl,
    "Neighborhood": neighborhood, "MSZoning": zoning, "BldgType": bldg, "LotArea": lot_area,
    "YrSold": yr_sold, "1stFlrSF": first, "2ndFlrSF": second, "GrLivArea": gr_liv,
    "TotalBsmtSF": bsmt, "YearBuilt": year_built, "YearRemodAdd": year_remod,
    "OverallQual": overall_qual, "OverallCond": overall_cond, "KitchenQual": kitchen,
    "ExterQual": exter, "CentralAir": "Y" if central_air else "N", "BedroomAbvGr": bedrooms,
    "TotRmsAbvGrd": rooms, "FullBath": full_bath, "HalfBath": half_bath,
    "GarageCars": garage_cars, "GarageArea": garage_area, "Fireplaces": fireplaces,
    "HouseStyle": "2Story" if second > 0 else "1Story",
}
# Keep dependent fields consistent with the inputs.
if bsmt == 0:
    house.update(BsmtFinSF1=0, BsmtFinSF2=0, BsmtUnfSF=0, BsmtFullBath=0, BsmtHalfBath=0,
                 BsmtQual=None, BsmtCond=None, BsmtExposure=None, BsmtFinType1=None, BsmtFinType2=None)
else:
    fin = min(tpl["BsmtFinSF1"], bsmt)
    house.update(BsmtFinSF1=fin, BsmtFinSF2=0, BsmtUnfSF=bsmt - fin)
if garage_cars == 0 or garage_area == 0:
    house.update(GarageCars=0, GarageArea=0, GarageType=None, GarageFinish=None,
                 GarageQual=None, GarageCond=None, GarageYrBlt=None)
else:
    house["GarageYrBlt"] = year_built
house["FireplaceQu"] = "Gd" if fireplaces > 0 else None

errors = []
if year_built > yr_sold:
    errors.append("Tahun dibangun tidak boleh setelah tahun terjual.")
if year_remod < year_built:
    errors.append("Tahun renovasi tidak boleh sebelum tahun dibangun.")
if first == 0:
    errors.append("Luas lantai 1 harus lebih dari 0.")

st.divider()
if errors:
    for e in errors:
        st.error(e)
    st.stop()

X_new = pd.DataFrame([house])[meta["raw_features"]]
price = float(pipe.predict(X_new)[0])
rmse = cv["RMSE USD (CV)"]
pct = int(np.searchsorted(ds["price_quantiles"], price).clip(0, 100))

st.subheader("Hasil estimasi")
r1, r2, r3 = st.columns(3)
r1.metric(f"Estimasi harga ({meta['currency']})", usd(price))
r2.metric("Rentang kasar (± 1 RMSE)", f"{usd(max(price - rmse, 0))} – {usd(price + rmse)}")
r3.metric("Posisi di data latih", f"Persentil ke-{pct}",
          help="Persentase rumah di data latih yang harganya di bawah estimasi ini.")
st.caption(f"Harga dalam dolar AS, nominal tahun {yr_sold} di Ames, Iowa. "
           f"Median data latih: {usd(ds['price_median'])}. "
           f"Harga per sq ft hunian: {usd(price / gr_liv)} (≈ {usd(price / (gr_liv * SQFT_TO_M2))}/m²).")

# ---------- Out-of-scope warnings ----------
checks = {"Luas lantai 1": ("1stFlrSF", first), "Luas lantai 2": ("2ndFlrSF", second),
          "Luas basement": ("TotalBsmtSF", bsmt), "Luas tanah": ("LotArea", lot_area),
          "Tahun dibangun": ("YearBuilt", year_built), "Luas garasi": ("GarageArea", garage_area),
          "Kamar tidur": ("BedroomAbvGr", bedrooms), "Total ruangan": ("TotRmsAbvGrd", rooms)}
outside = []
for label, (col, val) in checks.items():
    r = ranges[col]
    if val < r["min"] or val > r["max"]:
        outside.append(f"**{label}** ({val:,}) di luar rentang data latih {r['min']:,.0f}–{r['max']:,.0f}")
    elif (val < r["p01"] or val > r["p99"]) and val != 0:
        outside.append(f"**{label}** ({val:,}) termasuk 1% nilai paling ekstrem di data latih")
if price > ds["price_max"] or price < ds["price_min"]:
    outside.append(f"Estimasi berada di luar rentang harga data latih "
                   f"({usd(ds['price_min'])}–{usd(ds['price_max'])})")
if outside:
    st.warning("Estimasi kurang andal karena input berada di luar cakupan data:\n\n- "
               + "\n- ".join(outside))
else:
    st.success("Semua input berada dalam cakupan data latih.")

with st.expander("Detail: baris data yang dikirim ke pipeline"):
    st.caption("Fitur yang tidak diisi memakai nilai rumah tipikal (median/modus) dari data latih. "
               "Pipeline yang sama dengan saat pelatihan menjalankan feature engineering, "
               "imputasi, scaling, dan one-hot encoding.")
    st.dataframe(X_new.T.rename(columns={0: "nilai"}).astype(str), height=300)
