#!/usr/bin/env python3
"""
Build web/readyation_console.html from the template and READYation's replay file.

  python granite_ttm.py OMNI.csv out --finetune --replay      # writes out/readyation_replay.json
  python web/build_console.py out/readyation_replay.json      # writes web/readyation_console.html

The page is self-contained (data embedded), so it opens offline in any browser.
"""
import argparse, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '</head>\n<body style="margin:0">\n')
TAIL = '\n</body>\n</html>\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", help="readyation_replay.json from granite_ttm.py --replay")
    ap.add_argument("--template", default=os.path.join(HERE, "console_template.html"))
    ap.add_argument("--out", default=os.path.join(HERE, "readyation_console.html"))
    a = ap.parse_args()
    with open(a.replay, encoding="utf-8") as f:
        data = json.load(f)
    need = {"thr_6h_log10", "storms"}
    if not need <= data.keys() or not all("fc" in s for s in data["storms"]):
        raise SystemExit("replay file has no forecasts: run granite_ttm.py with --finetune --replay")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(a.template, encoding="utf-8") as f:
        page = f.read().replace("__DATA__", payload)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(HEAD + page + TAIL)
    print(f"wrote {a.out}  ({len(data['storms'])} storms, {len(data.get('watches', []))} no-storm watches, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
