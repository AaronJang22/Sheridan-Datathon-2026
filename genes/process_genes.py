#!/usr/bin/env python3
"""
Part 1 gene panel: extract radiation-response genes from the NASA GeneLab OSD-118 expression table.

NOTE: reconstructed after the fact. The original processing script was lost; this script was written
to reproduce data/processed_gene_dataset.csv and is checked byte-for-byte against it (--check).

Input  data/all_genes.csv : Illumina BeadChip AVG_Signal for 47,322 probes x 16 samples
                            (columns <chip>_<position>.AVG_Signal from three chips), blank column, ILMN_GENE.
Output data/processed_gene_dataset.csv : for each gene below, a header row, every probe row for that gene
                            (in input order), a short note on the first row, and two blank rows between genes.
                            The chip prefix is dropped from the headers (e.g. 9703627026_C.AVG_Signal -> C.AVG_Signal).

It also prints the per-gene value used in the report: the mean signal across the 16 samples of the
gene's strongest probe (the probe with the highest mean).

Usage: python process_genes.py [--raw data/all_genes.csv] [--out data/processed_gene_dataset.csv] [--check]
"""
import argparse, os, re, statistics

# gene -> note (the original file's notes, in Korean; English in the comment)
GENES = {
    "TP53":   "종양 억제 및 방사선 유도 세포 사멸 센서",           # tumour suppressor; radiation-induced cell-death sensor
    "CDKN1A": "p21; DNA 손상 시 세포 주기 정지 마커",               # p21; cell-cycle arrest marker after DNA damage
    "ATM":    "방사선 DNA 이중가닥 절단 감지",                      # senses radiation DNA double-strand breaks
    "IL6":    "급성 전신 염증 사이토카인",                          # acute systemic inflammatory cytokine
    "ICAM1":  "혈뇌장벽(BBB) 및 혈관 내피 염증 반응",               # blood-brain barrier / endothelial inflammation
    "SOD2":   "우주 방사선 산화 스트레스 제거 효소",                # mitochondrial enzyme clearing radiation oxidative stress
    "BAX":    "세포 자살/Apoptosis 촉진 인자",                      # promotes apoptosis
}
NL = "\r\n"          # the original file uses Windows line endings and no final newline
WIDTH = 20           # 16 samples, blank, ILMN_GENE, blank, note


def read_raw(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:] if l]
    return header, rows


def build(header, rows):
    samples = [re.sub(r"^\d+_", "", h) for h in header[:16]]          # drop BeadChip prefix
    head = ",".join(samples + ["", "ILMN_GENE", "", ""])
    blank = "," * (WIDTH - 1)
    blocks, probes = [], {}
    for gene, note in GENES.items():
        hits = [r for r in rows if r[-1] == gene]
        if not hits:
            raise SystemExit(f"{gene} not found in the input")
        probes[gene] = [[float(v) for v in r[:16]] for r in hits]
        out = [head]
        for i, r in enumerate(hits):
            out.append(",".join(r[:16] + ["", gene, "", note if i == 0 else ""]))
        blocks.append(NL.join(out))
    return (NL + blank + NL + blank + NL).join(blocks), probes


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(here, "data", "all_genes.csv"))
    ap.add_argument("--out", default=os.path.join(here, "data", "processed_gene_dataset.csv"))
    ap.add_argument("--check", action="store_true", help="compare with the existing output instead of writing it")
    a = ap.parse_args()

    text, probes = build(*read_raw(a.raw))
    if a.check:
        with open(a.out, "rb") as f:
            same = f.read() == text.encode("utf-8")
        print("matches", a.out, "byte for byte" if same else "-- DIFFERENT")
    else:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print("wrote", a.out)

    print("\nStrongest probe per gene (mean AVG_Signal across 16 samples), as used in the report:")
    print(f"  {'gene':7s} {'probes':>6s} {'strongest':>10s}")
    for gene in sorted(probes, key=lambda g: -max(statistics.mean(p) for p in probes[g])):
        best = max(statistics.mean(p) for p in probes[gene])
        print(f"  {gene:7s} {len(probes[gene]):6d} {best:10.0f}")


if __name__ == "__main__":
    main()
