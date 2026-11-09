"""Assemble the 30-province, 2011-2021 panel used in the study.

Every input is a published statistical release; nothing is imputed or simulated.
  * CEADs provincial energy inventories (sector x fuel, physical units)
  * CEADs apparent CO2 emissions, NBS GDP / GDP index / secondary value added /
    population / urban population / fiscal expenditure, invention patent grants,
    PKU digital financial inclusion index (all via the processed CUC panel)
  * NBS technology-market turnover (KarlHeinrich provincial panel)
  * provincial-capital coordinates and first-order contiguity
  * NBS sectoral value added 2019-2022 (validation of the service proxy only)
Usage: python build_panel.py <data_root> <out_dir>
"""
import sys, glob, os, re
import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
CUC = os.path.join(ROOT, "statistical-modeling_CUC_2026_PYH")
KH = glob.glob(os.path.join(ROOT, "China-Provincial-Panel-Dataset*", "data.xlsx"))[0]
PE = os.path.join(ROOT, "pe", "data", "raw", "nbs_exports")

EN2CN = {"Beijing": "北京", "Tianjin": "天津", "Hebei": "河北", "Shanxi": "山西",
         "InnerMongolia": "内蒙古", "Liaoning": "辽宁", "Jilin": "吉林", "Heilongjiang": "黑龙江",
         "Shanghai": "上海", "Jiangsu": "江苏", "Zhejiang": "浙江", "Anhui": "安徽", "Fujian": "福建",
         "Jiangxi": "江西", "Shandong": "山东", "Henan": "河南", "Hubei": "湖北", "Hunan": "湖南",
         "Guangdong": "广东", "Guangxi": "广西", "Hainan": "海南", "Chongqing": "重庆", "Sichuan": "四川",
         "Guizhou": "贵州", "Yunnan": "云南", "Shaanxi": "陕西", "Gansu": "甘肃", "Qinghai": "青海",
         "Ningxia": "宁夏", "Xinjiang": "新疆"}
CN2EN = {v: k for k, v in EN2CN.items()}
CN2EN["内蒙古"] = "Inner Mongolia"

# China Energy Statistical Yearbook 2022, Appendix IV (physical unit -> 10^4 tce)
TCE = {"Raw_Coal": 0.7143, "Cleaned_Coal": 0.9000, "Other_Washed_Coal": (0.2857 + 0.4286) / 2,
       "Briquettes": 0.7143, "Coke": 0.9714, "Other_Coking_Products": 1.1429, "Crude_Oil": 1.4286,
       "Gasoline": 1.4714, "Kerosene": 1.4714, "Diesel_Oil": 1.4571, "Fuel_Oil": 1.4286,
       "LPG": 1.7143, "Refinery_Gas": 1.5714, "Other_Petroleum_Products": 1.4286,
       "Coke_Oven_Gas": ((0.5714 + 0.6143) / 2) * 10, "Other_Gas": 0.3571 * 10,
       "Natural_Gas": ((1.1000 + 1.3300) / 2) * 10, "Heat": 0.03412, "Electricity": 0.1229 * 10,
       "Other_Energy": 1.0}
COAL = ["Raw_Coal", "Cleaned_Coal", "Other_Washed_Coal", "Briquettes", "Coke", "Coke_Oven_Gas",
        "Other_Coking_Products"]
FIRST_MFG, LAST_MFG = "Food Processing", "Scrap and waste"
HIGHTECH = ["Medical and Pharmaceutical Products", "Electronic and Telecommunications Equipment",
            "Instruments, Meters Cultural and Office Machinery", "Transportation Equipment",
            "Electric Equipment and Machinery", "Equipment for Special Purpose", "Ordinary Machinery"]


def energy_block(f):
    rows = []
    x = pd.ExcelFile(f)
    for s in x.sheet_names:
        m = re.match(r"([A-Za-z]+)(\d{4})$", s)
        if not m:
            continue
        prov, yr = EN2CN[m.group(1)], int(m.group(2))
        d = x.parse(s, header=0)
        d = d.rename(columns={d.columns[0]: "sector"})
        d = d.iloc[1:].copy()                       # drop unit row
        d["sector"] = d["sector"].astype(str).str.strip()
        stop = d.index[d["sector"] == "Rural"]
        d = d.loc[: stop[0]] if len(stop) else d
        for c in TCE:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0) * TCE[c]
        d["tce"] = d[list(TCE)].sum(axis=1)
        d["coal"] = d[COAL].sum(axis=1)
        sec = d["sector"].tolist()
        i0, i1 = sec.index(FIRST_MFG), sec.index(LAST_MFG)
        mfg = d.iloc[i0:i1 + 1]
        tot = d[d["sector"] == "Total Final Consumption"].iloc[0]
        rows.append(dict(province=prov, year=yr, en_total=tot["tce"], coal_total=tot["coal"],
                         en_mfg=mfg["tce"].sum(), coal_mfg=mfg["coal"].sum(),
                         en_hitech=mfg[mfg["sector"].isin(HIGHTECH)]["tce"].sum(),
                         en_tsp=d[d["sector"].str.startswith("Transport, Storage")]["tce"].sum()))
    return rows


rows = []
for f in sorted(glob.glob(os.path.join(CUC, "data_raw", "energy", "*.xlsx"))):
    rows += energy_block(f)
en = pd.DataFrame(rows)

base = pd.read_csv(os.path.join(CUC, "data_processed", "panel_data.csv"))
kh = pd.read_excel(KH).rename(columns={"年份": "year", "省份": "province",
                                       "Technology market turnover": "tech_turnover",
                                       "Total exports and imports": "trade",
                                       "Population density": "pop_density"})
kh = kh[["province", "year", "tech_turnover", "trade", "pop_density"]]

df = base.merge(en, on=["province", "year"], how="left").merge(kh, on=["province", "year"], how="left")
df = df[(df.year >= 2011) & (df.year <= 2021)].copy()   # 2022 technology-market series is corrupted
df["province_en"] = df.province.map(CN2EN)

# consistency between the independently computed totals and the CUC totals
df["chk_en"] = df.en_total / df.total_energy_tce
df["chk_coal"] = df.coal_total / df.coal_related_tce

# geography
co = pd.read_csv(os.path.join(CUC, "data_raw", "spatial", "province_capital_coordinates.csv"))
co = co[["province", "longitude", "latitude"]]
df = df.merge(co, on="province", how="left")
hz_lon, hz_lat = 120.153576, 30.287459   # Hangzhou city centre


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


df["dist_hz_km"] = haversine(df.longitude, df.latitude, hz_lon, hz_lat)
reg = pd.read_csv(os.path.join(CUC, "data_raw", "region_groups.csv"))
df = df.merge(reg, on="province", how="left")
df.to_csv(os.path.join(OUT, "panel_raw.csv"), index=False, encoding="utf-8")

w = pd.read_csv(os.path.join(CUC, "data_raw", "spatial", "spatial_weights.csv"))
w.to_csv(os.path.join(OUT, "contiguity_edges.csv"), index=False, encoding="utf-8")

# validation series: NBS producer-service value added 2019-2022
val = []
for f in glob.glob(os.path.join(PE, "分省年度数据*.csv")):
    yr = re.search(r"(20\d\d)", os.path.basename(f))
    if not yr or int(yr.group(1)) > 2022:
        continue
    raw = open(f, "rb").read().decode("utf-8-sig", errors="ignore")
    lines = [l for l in raw.splitlines() if l.strip()]
    hdr_i = [i for i, l in enumerate(lines) if l.startswith("地区")]
    if not hdr_i:
        continue
    hdr = [h.strip() for h in lines[hdr_i[0]].split(",")]
    for l in lines[hdr_i[0] + 1:]:
        cells = [c.strip() for c in l.split(",")]
        if len(cells) < 10 or not cells[0]:
            continue
        r = dict(zip(hdr, cells))
        try:
            val.append(dict(province=re.sub("(维吾尔自治区|壮族自治区|回族自治区|自治区|省|市)$", "", cells[0]),
                            year=int(yr.group(1)),
                            gdp=float(r[[h for h in hdr if h.startswith("地区生产总值")][0]]),
                            va_transport=float(r[[h for h in hdr if h.startswith("交通运输")][0]]),
                            va_finance=float(r[[h for h in hdr if h.startswith("金融业")][0]]),
                            va_wholesale=float(r[[h for h in hdr if h.startswith("批发和零售")][0]]),
                            va_other=float(r[[h for h in hdr if h.startswith("其他行业")][0]])))
        except (ValueError, IndexError, KeyError):
            continue
pd.DataFrame(val).to_csv(os.path.join(OUT, "validation_va.csv"), index=False, encoding="utf-8")
print(df.shape, df.province.nunique(), df.year.min(), df.year.max())
print("energy total ratio vs CUC: min %.4f max %.4f" % (df.chk_en.min(), df.chk_en.max()))
print("coal total ratio vs CUC:   min %.4f max %.4f" % (df.chk_coal.min(), df.chk_coal.max()))
print(df[["en_mfg", "en_total", "tech_turnover", "digital_finance", "dist_hz_km"]].describe().round(2).to_string())
print("validation rows:", len(val))
