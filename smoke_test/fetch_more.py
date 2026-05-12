"""fetch additional candidate images for instruments we're missing"""
import json, os, time, urllib.parse, requests

OUT = os.path.join(os.path.dirname(__file__), "candidates")
os.makedirs(OUT, exist_ok=True)

HEADERS = {"User-Agent": "Deep24-smoke-test/1.0 (research)"}

def get_with_retry(url, params, tries=5):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(8 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError:
            if i == tries - 1: raise
            time.sleep(5 * (i + 1))

def search(query, n=8):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {"action": "query", "format": "json", "list": "search",
              "srsearch": query + " filetype:bitmap", "srnamespace": "6", "srlimit": n}
    r = get_with_retry(url, params)
    return [h["title"] for h in r.json()["query"]["search"]]

def info(titles):
    url = "https://commons.wikimedia.org/w/api.php"
    params = {"action": "query", "format": "json", "titles": "|".join(titles),
              "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": "1024"}
    r = get_with_retry(url, params)
    out = []
    for pid, p in r.json()["query"]["pages"].items():
        if "imageinfo" not in p: continue
        ii = p["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        out.append({"title": p["title"],
                    "url": ii.get("thumburl") or ii["url"],
                    "description": (meta.get("ImageDescription", {}).get("value", "") or "")[:400],
                    "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(p["title"])})
    return out

def download(item, slot):
    safe = item["title"].replace(" ", "_").replace("File:", "")[:60]
    if "." not in safe: safe += ".jpg"
    fname = f"{slot:03d}_{safe}"
    path = os.path.join(OUT, fname)
    if os.path.exists(path): return path
    for tr in range(4):
        r = requests.get(item["url"], headers=HEADERS, timeout=60)
        if r.status_code == 429:
            time.sleep(8 * (tr + 1)); continue
        r.raise_for_status()
        with open(path, "wb") as f: f.write(r.content)
        return path

# focused queries
QUERIES = [
    ("voltmeter", "analog panel voltmeter dial display"),
    ("ammeter", "panel ammeter analog needle"),
    ("multimeter_analog", "analog multimeter"),
    ("pressure_gauge_psi", "tire pressure gauge psi analog"),
    ("dial_thermometer", "dial thermometer outdoor garden"),
    ("oven_thermometer", "oven thermometer dial"),
    ("kitchen_scale_dial", "kitchen scale spring dial"),
    ("postage_scale", "postal scale analog"),
    ("manometer_gauge", "manometer industrial dial face needle"),
    ("scale_butcher", "hanging scale butcher dial"),
    ("compass_bezel", "magnetic compass needle bearing"),
    ("hygrometer", "hygrometer humidity dial analog"),
]

candidates = []
seen = set()
for slot, (label, q) in enumerate(QUERIES):
    base = 200 + slot * 10
    print(f"[{label}] {q}")
    try:
        titles = search(q, n=4)
        items = info(titles) if titles else []
        for j, it in enumerate(items[:2]):
            if it["title"] in seen: continue
            seen.add(it["title"])
            try:
                path = download(it, base + j)
                if path:
                    print(f"  -> {os.path.basename(path)}")
                    candidates.append({"label": label, **it, "path": path})
            except Exception as e:
                print(f"  dl fail: {e}")
        time.sleep(2.5)
    except Exception as e:
        print(f"  fail: {e}")
        time.sleep(5)

with open(os.path.join(os.path.dirname(__file__), "candidates2.json"), "w") as f:
    json.dump(candidates, f, indent=2)
print(f"{len(candidates)} added")
