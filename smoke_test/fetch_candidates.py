"""fetch candidate instrument images from wikimedia commons"""
import json, os, time, urllib.parse, requests, sys

OUT = os.path.join(os.path.dirname(__file__), "candidates")
os.makedirs(OUT, exist_ok=True)

HEADERS = {"User-Agent": "Deep24-smoke-test/1.0 (research; contact deep24)"}

def get_with_retry(url, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 5 * (i + 1)
                print(f"  429, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
    raise RuntimeError("retry exhausted")

def search(query, n=8):
    """search commons for images, return file titles"""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query + " filetype:bitmap",
        "srnamespace": "6",  # File namespace
        "srlimit": n,
    }
    r = get_with_retry(url, params)
    return [hit["title"] for hit in r.json()["query"]["search"]]

def info(titles):
    """get download URL + description for files"""
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": "1024",
    }
    r = get_with_retry(url, params)
    pages = r.json()["query"]["pages"]
    out = []
    for pid, p in pages.items():
        if "imageinfo" not in p:
            continue
        ii = p["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        desc = meta.get("ImageDescription", {}).get("value", "")
        out.append({
            "title": p["title"],
            "url": ii.get("thumburl") or ii["url"],
            "description": desc,
            "page": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(p["title"]),
        })
    return out

def download(item, slot):
    """download to candidates/ with slot prefix"""
    safe = item["title"].replace(" ", "_").replace("File:", "")[:60]
    ext = safe.rsplit(".", 1)[-1].lower()
    fname = f"{slot:02d}_{safe}"
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        return path
    for tr in range(4):
        try:
            r = requests.get(item["url"], headers=HEADERS, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (tr + 1))
                continue
            r.raise_for_status()
            break
        except requests.exceptions.HTTPError:
            if tr == 3:
                raise
            time.sleep(3 * (tr + 1))
    with open(path, "wb") as f:
        f.write(r.content)
    return path

QUERIES = [
    ("pressure_gauge", "manometer pressure gauge analog"),
    ("pressure_gauge2", "boiler pressure gauge bar"),
    ("thermometer", "analog thermometer dial outdoor"),
    ("thermometer2", "mercury thermometer celsius"),
    ("kitchen_scale", "kitchen analog scale dial"),
    ("bathroom_scale", "bathroom analog scale weighing"),
    ("voltmeter", "analog voltmeter dial galvanometer"),
    ("ammeter", "analog ammeter dial amperes"),
    ("speedometer", "speedometer analog dashboard"),
    ("fuel_gauge", "fuel gauge analog dashboard"),
    ("watch", "wristwatch analog dial"),
    ("blood_pressure", "sphygmomanometer aneroid"),
    ("lab_balance", "analytical balance dial"),
    ("barometer", "aneroid barometer dial"),
]

candidates = []
for slot, (label, q) in enumerate(QUERIES):
    print(f"[{label}] search: {q}")
    try:
        titles = search(q, n=4)
        items = info(titles) if titles else []
        for j, it in enumerate(items[:2]):
            try:
                path = download(it, slot * 10 + j)
                print(f"  -> {path}")
                candidates.append({
                    "label": label,
                    "title": it["title"],
                    "url": it["url"],
                    "page": it["page"],
                    "path": path,
                    "description": (it["description"] or "")[:300],
                })
            except Exception as e:
                print(f"  download fail: {e}")
        time.sleep(1.5)
    except Exception as e:
        print(f"  search fail: {e}")

with open(os.path.join(os.path.dirname(__file__), "candidates.json"), "w") as f:
    json.dump(candidates, f, indent=2)
print(f"\n{len(candidates)} candidates saved")
