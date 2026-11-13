# CDCL-GPTQ su un modello reale: TinyLlama-1.1B

Misura del 2026-11-13 su questa macchina (Intel Core Ultra 9 285K, 24 core; RTX 5070 Ti usata solo per
catturare le Hessiane e per valutare). Rust 1.100 + MSVC Build Tools 2026, `RUSTFLAGS="-C target-cpu=native"`.
Questa è la prima verifica su un LLM reale: prima di ora c'erano solo i layer sintetici di `cargo bench`.

## Protocollo

| | |
|---|---|
| modello | `TinyLlama/TinyLlama_v1.1` — 1,100 M parametri, architettura Llama, GQA (4 kv head), 22 layer |
| pesi di partenza | convertiti in fp16 safetensors: **2200 MB** (2098 MiB) |
| calibrazione | WikiText-2 **train**, 128 sequenze × 2048 token = 262 144 token (`scripts/capture_hessians.py`, GPU, ~3 min) |
| Hessiane | `hess-128x2048.safetensors`: 88 matrici + 66 layer in *alias* (q/k/v e gate/up condividono l'input) = 3,7 GB |
| layer quantizzati | 154 tensori 2-D = **969 M parametri**; restano fp16 `embed_tokens` e `lm_head` (65,5 M ciascuno), layer norm e bias |
| valutazione | WikiText-2 **test**, 334 521 token in chunk non sovrapposti da 1024 (`scripts/eval_ppl.py`): perplexity + accuratezza top-1 next-token |
| confronto | GPTQ e CDCL con **impostazioni identiche** (act-order, damp 0.01, group indicato), pesi fake-quant (`out/dequant`) caricati al posto degli originali |

## Risultati

| configurazione | bpw effettivo | modello (MB) | vs fp16 | tempo quant (s) | gain loss proxy | perplexity | Δppl vs fp16 | top-1 % | Δppl vs GPTQ |
|---|---|---|---|---|---|---|---|---|---|
| **fp16 (riferimento)** | 16.0 | 2200 | 1.00x | — | — | **8.72** | — | 54.18 | — |
| 4 bit, g128, gptq | 4.19 | 771 | 2.85x | 78 | — | 9.10 | +4.4% | 53.55 | — |
| 4 bit, g128, cdcl | 4.19 | 771 | 2.85x | 235 | −8.3% | **9.06** | +4.0% | 53.62 | −0.4% |
| 3 bit, g128, mse-grid, gptq | 3.19 | 650 | 3.38x | 109 | — | 12.88 | +47.8% | 49.08 | — |
| 3 bit, g128, mse-grid, cdcl | 3.19 | 650 | 3.38x | 260 | −8.2% | **10.75** | +23.3% | 50.84 | **−16.6%** |
| 3 bit, g128, gptq | 3.19 | 650 | 3.38x | 55 | — | 13.67 | +56.7% | 47.94 | — |
| 3 bit, g128, cdcl | 3.19 | 650 | 3.38x | 234 | −13.9% | **12.44** | +42.6% | 49.04 | **−9.0%** |
| 2 bit, g64, mse-grid, gptq | 2.38 | 552 | 3.99x | 92 | — | 197.9 | +2170% | 22.32 | — |
| 2 bit, g64, mse-grid, cdcl | 2.38 | 552 | 3.99x | 257 | −10.4% | **47.2** | +442% | 33.49 | **−76.1%** |
| 2 bit, g128, gptq | 2.19 | 529 | 4.16x | 69 | — | 2,9e+04 | — | 0.34 | — |
| 2 bit, g128, cdcl | 2.19 | 529 | 4.16x | 250 | −37.8% | 1671 | — | 7.66 | −94.2% |

**bpw effettivo** comprende i metadati per gruppo (scala f16 + zero 8 bit): a 3 bit g128 sono 3.00 bit di
codice + 0.19 di metadati. Il rapporto **vs fp16** è sul file `packed` completo, quindi include anche embedding,
lm_head, norm e bias rimasti in fp16: sono ~131 M parametri ≈ 262 MB e sono il collo di bottiglia del numero
finale. Sui soli pesi quantizzati il guadagno è 16/3.19 = **5,0×** a 3 bit (e il file `dequant` è infatti
sempre 2200 MB, identico all'originale).

Lettura:

- **4 bit:** CDCL vale quasi niente sul ppl (−0.4%) — a 4 bit GPTQ è già vicino al pavimento, come previsto.
- **3 bit:** è la zona utile. La loss proxy scende del 13,9% e la perplexità del **9,0%** a parità di bit, group
  e impostazioni (top-1 +1,1 punti). Con `--mse-grid` il vantaggio sale al **16,6%** (12.88 → 10.75): sul
  mondo reale la griglia MSE scelta per gruppo si sposa meglio con la riparazione di quanto facesse in sintetico.
- **2 bit:** la differenza è qualitativa. GPTQ a 2 bit g128 produce un modello morto (ppl 29 000, top-1 0,34%);
  CDCL lo porta a 1671 (ancora ingestibile) e con g64 + mse-grid a **47 ppl**, dove il testo in inglese è ancora
  sintatticamente corretto (198 per GPTQ). La perdita di loss proxy (−37,8%) non si traduce in proporzionalità
  sul ppl: a 2 bit contano di più group e griglia che la riparazione.
- **Costo:** 3–4× il tempo di GPTQ ai default (234–260 s vs 55–109 s per 969 M di pesi su 24 core CPU).

## Il modello funziona ancora?

Sì, fino a 3 bit. Oltre ai numeri: generazione con seed fisso, temperature 0.8, top-p 0.95, stesso prompt
(TinyLlama base è ripetitivo anche in fp16):

| | "The capital of France is" |
|---|---|
| fp16 | "…the The capital of Paris is the The capital of the world is the capital world is the capital…" |
| 3 bit mse cdcl (3.38× più piccolo) | "…divided with the capital of China, where the capital of Vietnam and the capital of Japan. and the capital of Hong Kong are divided into three countries: Indonesia, China, and Japan." |
| 2 bit g64 mse cdcl (3.99× più piccolo) | "…the royal of France is the royal of the capital of France is the royal of the royal…" (sintassi ok, semantica in collasso) |
| 2 bit g128 cdcl | "…the The United States.." poi token spazzatura → non usabile |

Verifica di integrità del formato compatto: `cdclq dequant` su `runs/b3-cdcl/packed/model.safetensors` restituisce
gli **stessi 201 tensori, stessi dtype e forme, differenza massima 0.0** rispetto a `runs/b3-cdcl/dequant/model.safetensors`.
Quindi il `packed` non perde nulla rispetto al modello fake-quant valutato qui sopra.

## Cosa non va ancora bene (dal run reale)

- `scripts/capture_hessians.py` funziona (mai eseguito prima): 88 Hessiane + 66 alias corretti, q/k/v e gate/up
  condivisi davvero.
- Il rapporto di compressione complessivo è frenato da embedding + lm_head in fp16: per scendere sotto ~3,4× a
  3 bit servono pesi non lineari quantizzati (o lm_head quantizzato: basta passarlo nelle Hessiane con `--skip ""`).
- 2 bit g128 resta irrealisticamente distruttivo: il group 64 è obbligatorio, e servirebbe un budget di bit
  non uniforme per layer (i layer 0–2 sono i più sensibili).
- `hess-128x2048.safetensors` pesa 3,7 GB per un modello da 1,1 B: con `d_in` grandi (MLP) domina la memoria
  del disco, e per un modello 7 B servirebbe catturare a pezzi con `--only` (come già documentato).

## Come riprodurre

```powershell
# 0) ambiente
$env:RUSTFLAGS="-C target-cpu=native"; cargo build --release
#    transformers 4.52 richiede huggingface-hub < 1.0 (l'ambiente di sistema aveva 1.33) -> virtualenv:
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install --no-deps "huggingface-hub==0.34.4"

# 1) modello + testi (il repo TinyLlama ha solo pytorch_model.bin fp32: convertito in model.safetensors fp16)
#    data\calib.txt e data\wikitext2-test.txt da Salesforce/wikitext, config wikitext-2-raw-v1

# 2) Hessiane (GPU, ~3 min)
.venv\Scripts\python.exe scripts\capture_hessians.py --model models\TinyLlama-1.1B --text data\calib.txt `
       --out hess-128x2048.safetensors --nsamples 128 --seqlen 2048 --batch 2

# 3) griglia 4/3 bit x gptq/cdcl + valutazione ppl
powershell -File scripts\run_matrix.ps1
.\.venv\Scripts\python.exe scripts\summarize.py --md runs\table.md

# 4) singoli esperimenti + testo generato
.\target\release\cdclq.exe quantize --weights models\TinyLlama-1.1B\model.safetensors `
   --hessians hess-128x2048.safetensors --out-dir runs\b3-mse-cdcl --bits 3 --group 128 --mse-grid
.\.venv\Scripts\python.exe scripts\eval_ppl.py --model models\TinyLlama-1.1B --text data\wikitext2-test.txt `
   --dequant runs\b3-mse-cdcl\dequant\model.safetensors --json runs\b3-mse-cdcl.eval.json
.\.venv\Scripts\python.exe scripts\gen_compare.py --model models\TinyLlama-1.1B `
   --dequant runs\b3-mse-cdcl\dequant\model.safetensors --labels 3bit --seed 7
.\target\release\cdclq.exe dequant --input runs\b3-cdcl\packed\model.safetensors --output runs\b3-cdcl\roundtrip.safetensors
```

File prodotti (enormi, da non versionare): `runs/<config>/{report.json,packed,dequant}`, `runs/<config>.eval.json`,
`runs/<config>.log`, `hess-128x2048.safetensors` (3,7 GB), `models/TinyLlama-1.1B/model.safetensors` (2,2 GB).
`runs/*/roundtrip.safetensors` (2,2 GB) serve solo alla verifica di integrità.
