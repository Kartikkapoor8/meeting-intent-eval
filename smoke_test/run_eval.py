"""run claude opus 4.6 on each verified image, 8 samples each, T=1.0"""
import base64, json, os, sys, time
import anthropic

HERE = os.path.dirname(__file__)
GT = json.load(open(os.path.join(HERE, "ground_truth.json")))
IMG_DIR = os.path.join(HERE, "images")
OUT = os.path.join(HERE, "raw_samples.json")
USAGE_OUT = os.path.join(HERE, "usage.json")

MODEL = "claude-opus-4-6"
N_SAMPLES = 8
TEMP = 1.0
COST_CAP_USD = 5.0
# opus 4.6 pricing (per million tokens)
PRICE_IN = 15.0 / 1_000_000
PRICE_OUT = 75.0 / 1_000_000

SYSTEM_PROMPT = """You will be shown a photo of an analog instrument. Identify the instrument type and read its current value. Output ONLY valid JSON in this exact format:
{
  "instrument_type": "<type>",
  "reading": <number or string for clocks>,
  "unit": "<unit>",
  "scale_max": <number or null>,
  "confidence": "<high|medium|low>"
}

For clocks, reading should be "HH:MM" format.
For other instruments, reading should be a number.
Do not include any text outside the JSON."""

def encode_image(path):
    ext = path.rsplit(".", 1)[-1].lower()
    media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return media, base64.standard_b64encode(f.read()).decode()

def main():
    # use oauth token if ANTHROPIC_API_KEY is empty
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        client = anthropic.Anthropic()
    else:
        oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
        if not oauth:
            raise RuntimeError("no auth available")
        client = anthropic.Anthropic(auth_token=oauth, default_headers={"anthropic-beta": "oauth-2025-04-20"})
    raw = {}
    # resume if partial exists
    if os.path.exists(OUT):
        raw = json.load(open(OUT))
        print(f"resuming, {sum(len(v) for v in raw.values())} samples already done")
    total_in = total_out = 0
    cost = 0.0
    for entry in GT:
        img = entry["image"]
        path = os.path.join(IMG_DIR, img)
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        if img not in raw:
            raw[img] = []
        media, b64 = encode_image(path)
        needed = N_SAMPLES - len(raw[img])
        for s in range(needed):
            if cost >= COST_CAP_USD:
                print(f"COST CAP HIT at ${cost:.2f}, stopping")
                break
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=200,
                    temperature=TEMP,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                            {"type": "text", "text": "Read this instrument."}
                        ]
                    }]
                )
                text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens
                total_in += in_tok
                total_out += out_tok
                step_cost = in_tok * PRICE_IN + out_tok * PRICE_OUT
                cost += step_cost
                raw[img].append({"text": text, "in_tok": in_tok, "out_tok": out_tok, "cost": step_cost})
                print(f"{img} s{len(raw[img])}: {text[:80]}... [${cost:.3f}]")
                # save after every call
                with open(OUT, "w") as f:
                    json.dump(raw, f, indent=2)
            except Exception as e:
                print(f"  api error: {e}")
                time.sleep(3)
        if cost >= COST_CAP_USD:
            break
        time.sleep(0.5)

    with open(USAGE_OUT, "w") as f:
        json.dump({"total_in": total_in, "total_out": total_out, "cost_usd": cost,
                   "n_images": len(raw), "n_samples_total": sum(len(v) for v in raw.values())}, f, indent=2)
    print(f"\nDONE. cost=${cost:.3f}, in={total_in}, out={total_out}")

if __name__ == "__main__":
    main()
