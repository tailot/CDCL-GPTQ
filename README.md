# cdclq — CDCL-GPTQ (Modulo B, corretto e implementato in Rust)

Quantizzatore *post-training* dei pesi per LLM (safetensors → INT2–INT8 a gruppi), scritto in Rust.
Parte da GPTQ (Cholesky esatto, aggiornamenti lazy a blocchi) e poi **ripara i conflitti**: usa la
decomposizione esatta della loss per trovare i sotto-blocchi peggiori e li migliora con due mosse
esatte e monotone (discesa per coordinate sui codici interi + raffinamento in forma chiusa delle
scale), alternandole fino a convergenza.

**Risultato misurato su un LLM reale (TinyLlama-1.1B, WikiText-2, stessi bit, stessa griglia, stesso
ordinamento, stessa dampatura):** perplexity ridotta del **−9.0% a 3 bit** e **−16.6% a 3 bit con
`--mse-grid`** rispetto a GPTQ, a parità di compressione (3.19 bit/peso → modello 3,38× più piccolo), al costo
di **~3–4× il tempo** di GPTQ. A 2 bit la differenza è qualitativa: GPTQ produce un modello morto (ppl 29 000),
CDCL con `--group 64 --mse-grid` lo porta a 47. Su layer sintetico il guadagno era −6% … −17%; vedi "Risultati
su un LLM reale" e "Risultati" (sintetico).

## Cosa è stato corretto rispetto alla specifica

| # | Specifica | Problema | Qui |
|---|---|---|---|
| 1 | Pseudocodice: aggiornamenti con `H_inv` piena e statica | Non è GPTQ: manca l'aggiornamento di Schur dopo ogni colonna; le compensazioni risultano sbagliate | Fattore di Cholesky **superiore** `U` di `H⁻¹` (`H⁻¹ = UᵀU`) calcolato con il trucco del flip (1 Cholesky + 1 inversa triangolare). Test: `UᵀU·H = I`; i codici coincidono al **100%** con un GPTQ numpy float64 indipendente (`scripts/ref_gptq_check.py`) |
| 2 | Dopo 2 retry falliti si ripristina `checkpoint_W` e si esce | Il sotto-blocco resta **non quantizzato** (in virgola mobile) | Nessun retry né rollback: la riparazione è monotona e non può lasciare pesi non quantizzati |
| 3 | "Clausola" che forza l'arrotondamento opposto sul peso peggiore | Per costruzione *peggiora* quel peso; nessuna garanzia sul totale | Minimizzazione **esatta** per coordinata: il punto della griglia più vicino a `ŵⱼ + gⱼ/Hⱼⱼ` (gradiente esatto `g = eH`) |
| 4 | Conflitto = errore relativo vs `‖WX‖` | Segnale rumoroso, calcolato a parte, non lega al costo reale | Decomposizione **esatta** `L = Σⱼ εⱼ²`, `εⱼ = (w_cur − q)/Uⱼⱼ` (non negativa, gratis nel passaggio GPTQ, verificata da test) |
| 5 | `C_ij` per elemento con gradiente su token strutturali | Costoso (gradiente per elemento), maschera non strutturata (indici sparsi) | Criticità per **gruppo di colonne** `H_jj·‖W_:j‖²` (saliency OBS), gruppi interi a 8 bit. I token strutturali pesano nell'Hessiana (`--struct-weight` nello script di cattura). Il gradiente per elemento **non** è implementato |
| 6 | Propagazione cross-layer `X ← X + βΔY` | Il segno riporta l'input al valore FP: il layer successivo ottimizza come se il precedente fosse perfetto, senza compensare | Calibrazione **asimmetrica** esatta: `W* = W·M·H̃⁻¹` con `M = X X̃ᵀ` (equazioni normali). Nel motore e validata in sintetico; lo script di cattura **non** produce ancora `M` |
| 7 | "INT3" con 5% a INT8 | I bit effettivi non erano contati | Il report dà bit di codici e bit totali (scala f16 + zero 8 bit per gruppo e riga); lo scheletro alza i bit/peso e il benchmark lo segnala |
| 8 | Gruppi, scale, damping, batch update `...` | Non specificati | Implementati e configurabili |

## Algoritmo (per layer `W ∈ R^{d_out×d_in}`, `H = XXᵀ`)

1. **Ordinamento colonne:** gruppi "scheletro" protetti (se `--skeleton-pct > 0`), poi act-order.
2. **Fattorizzazione:** `U` con `H⁻¹ = UᵀU` (f64, blocchi, parallela). Condivisa tra layer che
   hanno lo stesso input (q/k/v, gate/up) quando l'ordinamento coincide.
3. **Passata GPTQ** a blocchi di 128 colonne con aggiornamento lazy (GEMM). Le righe sono
   indipendenti → parallelizzate con rayon. Fornisce `εⱼ²` per colonna.
4. **Conflitti:** i sotto-blocchi con loss maggiore (`--refine-frac`, default tutti).
5. **Riparazione** (`--passes`, arresto anticipato con `--tol`): alterna
   - **raffinamento scale:** con `u = q − z`, `Δs = (g·u)/(u H_CC uᵀ)` (forma chiusa, scala arrotondata a f16);
   - **discesa per coordinate** sui codici interi, con gradiente esatto del blocco.

   Ogni mossa non può aumentare la loss (testato: la loss tracciata coincide con quella calcolata
   direttamente, e `loss_final ≤ loss_gptq`). Ogni layer riporta anche `loss_gptq`: ciò che GPTQ
   avrebbe ottenuto con le stesse impostazioni.

## Installazione

Serve Rust (https://rustup.rs). Le dipendenze sono fissate in `Cargo.toml`/`Cargo.lock` (compatibili con Rust ≥ 1.75).

```bash
cargo build --release          # binario: target/release/cdclq (Windows: target\release\cdclq.exe)
cargo test --release           # 9 test
```

PowerShell, con ottimizzazioni per la CPU locale (facoltativo): `$env:RUSTFLAGS="-C target-cpu=native"; cargo build --release`.
Un binario Linux x86_64 già compilato è in `dist/linux-x86_64/cdclq`. I kernel GEMM usano AVX/FMA con
rilevamento a runtime.

## Uso

```bash
# 1) Hessiane di calibrazione (PyTorch + transformers; vedi il docstring per memoria e --only)
python scripts/capture_hessians.py --model ./modello --text calib.txt --out hess.safetensors \
       --nsamples 128 --seqlen 2048 [--struct-chars "(){}=<>" --struct-weight 4]

# 2) Quantizzazione
cdclq quantize --weights model.safetensors --hessians hess.safetensors --out-dir out \
       --bits 3 --group 128            # default: act-order, riparazione completa

# 3) Output
#   out/dequant/<shard>.safetensors  pesi dequantizzati, stesso dtype (per valutare in HF/lm-eval)
#   out/packed/<shard>.safetensors   modello bit-packed (qweight, scales f16, qzeros u8, perm, gbits, gsize)
#   out/report.json                  loss GPTQ vs finale, bit, tempi per layer
cdclq dequant --input out/packed/model.safetensors --output ricostruito.safetensors

# 4) valutazione (ppl + top-1 su testo held-out, pesi fake-quant) e testo generato
python scripts/eval_ppl.py --model ./modello --text test.txt --dequant out/dequant/model.safetensors \
       --json out.eval.json
python scripts/gen_compare.py --model ./modello --dequant out/dequant/model.safetensors --labels cdcl
python scripts/summarize.py            # tabella markdown da tutti i report.json + *.eval.json in runs/
```

Opzioni principali (`cdclq quantize --help`): `--bits`, `--group` (0 = una scala per riga),
`--damp`, `--no-act-order`, `--mse-grid`, `--skeleton-pct/--skeleton-bits`,
`--method gptq|cdcl` (per confronto), `--passes`, `--tol`, `--refine-frac`, `--no-refit-scales`,
`--only/--skip` (sottostringhe), `--threads`. Con più shard: `--weights a.safetensors b.safetensors`
(l'output mantiene i nomi dei file). Le Hessiane sono cercate come `<nome tensore senza .weight>.H`
(con alias da metadato `aliases`).

Profili di velocità: `--passes 2 --refine-frac 0.5` (veloce), `--passes 4` (bilanciato), default (completo).

## Risultati su un modello reale (TinyLlama-1.1B)

Prima verifica su un LLM reale (macchina: 24 core CPU, RTX 5070 Ti usata solo per Hessiane e valutazione).
Modello `TinyLlama/TinyLlama_v1.1` (1,1 B parametri, fp16, 2200 MB), calibrazione WikiText-2 train
(128 × 2048 token), valutazione WikiText-2 **test** (334 521 token, chunk da 1024, pesi fake-quant di
`out/dequant`). `--group 128 --damp 0.01`, act-order, GPTQ e CDCL con impostazioni identiche. Dettaglio completo,
campioni di testo generato e comandi di riproduzione: `runs/RESULTS-tinyllama.md`.

| configurazione | bit/peso | modello (MB) | vs fp16 | tempo (s) | perplexity | vs fp16 | top-1 % | vs GPTQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 (riferimento) | 16 | 2200 | 1.00× | — | **8.72** | — | 54.18 | — |
| 4 bit g128 GPTQ | 4.19 | 771 | 2.85× | 78 | 9.10 | +4.4% | 53.55 | — |
| 4 bit g128 CDCL | 4.19 | 771 | 2.85× | 235 | **9.06** | +4.0% | 53.62 | −0.4% |
| 3 bit g128 `--mse-grid` GPTQ | 3.19 | 650 | 3.38× | 109 | 12.88 | +47.8% | 49.08 | — |
| 3 bit g128 `--mse-grid` CDCL | 3.19 | 650 | 3.38× | 260 | **10.75** | +23.3% | 50.84 | **−16.6%** |
| 3 bit g128 GPTQ | 3.19 | 650 | 3.38× | 55 | 13.67 | +56.7% | 47.94 | — |
| 3 bit g128 CDCL | 3.19 | 650 | 3.38× | 234 | **12.44** | +42.6% | 49.04 | **−9.0%** |
| 2 bit g64 `--mse-grid` GPTQ | 2.38 | 552 | 3.99× | 92 | 197.9 | +2170% | 22.32 | — |
| 2 bit g64 `--mse-grid` CDCL | 2.38 | 552 | 3.99× | 257 | **47.2** | +442% | 33.49 | **−76.1%** |
| 2 bit g128 GPTQ | 2.19 | 529 | 4.16× | 69 | 2.9e+04 | — | 0.34 | — |
| 2 bit g128 CDCL | 2.19 | 529 | 4.16× | 250 | 1671 | — | 7.66 | −94.2% |

Cosa dice, onestamente:

- **Il modello funziona ancora** fino a 3 bit: top-1 50.8% contro 54.2% fp16 e testo generato ancora corretto
  (a volte più vario dell'fp16, che su TinyLlama base è molto ripetitivo). `packed → dequant` è bit-esatto.
- Il guadagno di CDCL **cresce al calare dei bit** (−0.4% → −9.0% → −76%): coerente con la sintesi, che a 3 bit
  prevedeva −9.6%. **A 4 bit invece vale quasi niente** (−0.4% contro il −6.9% sintetico): il layer sintetico
  sopravvaluta il guadagno quando GPTQ è già vicino al pavimento.
- Il rapporto complessivo (3.38× a 3 bit) è frenato da embedding + lm_head rimasti fp16 (≈262 MB): sui soli pesi
  quantizzati è 16/3.19 = 5,0×.
- 2 bit g128 non è usabile con nessuno dei due metodi; con g64 + `--mse-grid` CDCL resta in piedi (47 ppl).
  La perdita di loss proxy (−37.8%) non si traduce in proporzionalità sul ppl.
- Costo reale: **234–260 s contro 55–109 s** per 969 M di pesi quantizzati (24 core).

Script aggiunti per queste misure (non presenti nella specifica): `scripts/eval_ppl.py` (ppl/top-1 sui pesi
dequantizzati), `scripts/gen_compare.py` (testo generato fp16 vs quantizzato, stesso seed),
`scripts/summarize.py` (tabella dai `report.json` + `.eval.json`), `scripts/run_matrix.ps1` (griglia bit × metodo).

## Risultati

Layer sintetico 512×2048, 16384 token di calibrazione, 4096 **held-out**, 1 core, `--group 128`.
Attivazioni correlate, con canali outlier e token "massivi". Errore = `‖Wx − Ŵx‖² / ‖Wx‖²` su token mai visti.
Tutti i metodi condividono pesi, Hessiana, griglia, ordinamento e damping.

| 3 bit (3.19 bit/peso) | errore held-out | vs GPTQ | tempo |
|---|---:|---:|---:|
| RTN | 5.74e-2 | +826% | 0.01 s |
| **GPTQ** | 6.19e-3 | — | 0.46 s |
| solo raffinamento scale | 6.08e-3 | −1.9% | 0.83 s |
| solo discesa per coordinate | 6.15e-3 | −0.6% | 0.87 s |
| CDCL veloce (50% sotto-blocchi, 2 passate) | 5.99e-3 | −3.3% | 0.85 s |
| CDCL bilanciato (4 passate) | 5.72e-3 | −7.6% | 1.30 s |
| **CDCL default** (fino a 10 passate) | **5.60e-3** | **−9.6%** | 2.36 s |

Le due mosse da sole valgono poco; **insieme** (alternate) rendono molto: è l'alternanza codici ↔ scale
che produce il guadagno. La discesa per coordinate non basta, contrariamente a quanto la specifica lasciava supporre.

| Configurazione (held-out, vs GPTQ con le stesse impostazioni) | GPTQ | CDCL default | variazione |
|---|---:|---:|---:|
| 4 bit, g128 | 1.378e-3 | 1.284e-3 | **−6.9%** |
| 3 bit, g128 | 6.195e-3 | 5.602e-3 | **−9.6%** |
| 3 bit, g128, `--mse-grid` (anche per GPTQ) | 4.857e-3 | 4.264e-3 | **−12.2%** |
| 2 bit, g64 | 2.380e-2 | 1.968e-2 | **−17.3%** |

Altri esperimenti (stesso layer, 3 bit):

- **Scheletro 5% a 8 bit (3.5 bit/peso):** GPTQ+scheletro 5.23e-4, CDCL+scheletro 4.33e-4 (−17.2%).
  L'enorme calo rispetto a "senza scheletro" dipende dagli outlier sintetici e non è trasferibile ai modelli reali.
- **Input quantizzati (rumore relativo 0.08):** GPTQ simmetrico 1.330e-2 → GPTQ con target asimmetrico
  1.269e-2 (−4.6%) → CDCL con target asimmetrico 1.206e-2 (**−9.3%**).
- **Layer 512×4096** (6144 token, solo 1.5× la dimensione: calibrazione sovra-adattata, errore held-out
  ≈2× quello di calibrazione): CDCL default −6.2% in 10.5 s contro 3.1 s di GPTQ.

Riproduzione: `cdclq bench --rows 512 --cols 2048 --tokens 16384 --test-tokens 4096 [--bits 4] [--group 64] [--mse-grid] [--skeleton-pct 5] [--asym-noise 0.08]`.

## Verifiche eseguite

- 9 test unitari: Cholesky a blocchi, `UᵀU·H = I`, GEMM parallelo = seriale, pack/unpack a bit misti,
  decomposizione esatta della loss, monotonia + contabilità esatta della riparazione, permutazione/scheletro,
  caso asimmetrico `M = H ⇒ W* = W`.
- Confronto con un GPTQ numpy float64 indipendente: **codici identici al 100%**, loss relativa differente 0.
- Test end-to-end del CLI su un modello giocattolo (alias q/k/v, tensori 1-D e embedding passanti):
  nomi/dtype/forme invariati, `packed → dequant` **bit-esatto** rispetto all'output diretto, loss riportata
  uguale a un calcolo numpy indipendente (differenza relativa 3e-5).

## Limiti noti (da leggere)

- **Un solo modello reale misurato** (TinyLlama-1.1B, 1,1 B): il procedimento è in `runs/RESULTS-tinyllama.md`.
  Non misurati: modelli 7 B+, altri family (Mistral/Qwen), benchmark di downstream (lm-eval), e la pipeline
  sequenziale con input quantizzati. Su TinyLlama i guadagni ci sono (−9% … −17% a 3 bit) ma **a 4 bit sono
  trascurabili**; i numeri sintetici non vanno usati come previsione a bit alti.
- `scripts/capture_hessians.py` **eseguito e funzionante** su TinyLlama (88 Hessiane + 66 alias q/k/v e gate/up,
  `--skip lm_head` default). Cattura le Hessiane del modello FP (calibrazione simmetrica); non produce `M`.
  Serve un ambiente con `huggingface-hub < 1.0` (transformers 4.52 non funziona con hub 1.x).
- Il motore supporta `M`, ma una pipeline *sequenziale* (input quantizzati layer dopo layer) richiede un
  driver che esegua il modello quantizzato: non incluso.
- Il formato `packed` usa permutazione act-order e zero a 8 bit (24 bit di metadati per gruppo e riga): **non
  ci sono kernel di inferenza**. Per valutare/usare il modello oggi serve `out/dequant` (pesi fake-quant).
- Solo CPU. Tempo misurato su modello reale: **3–4× GPTQ** ai default (234–260 s vs 55–109 s per 969 M di pesi,
  24 core). Picco di memoria per layer ≈ 24·d_in² byte (d_in = 4096 → ~0.4 GB; 11008 → ~2.9 GB).
- Le Hessiane costano: `hess-128x2048.safetensors` = 3,7 GB per un modello da 1,1 B (e ~3,9 GB di VRAM in fp32
  durante la cattura). Su modelli grandi catturare a pezzi con `--only`.
- Il Modulo A (`lllayer`) non è toccato da questo progetto.

## Struttura

```
src/linalg.rs    GEMM parallelo, Cholesky a blocchi, inversa triangolare, U di H⁻¹
src/quant.rs     griglie, scale f16, bit-packing, QLayer
src/engine.rs    GPTQ + conflitti + riparazione + target asimmetrico (+ test)
src/pipeline.rs  shard, alias, report, packed → dequant
src/io.rs        safetensors (F32/F16/BF16), Hessian store
src/bench.rs     benchmark sintetico
scripts/         capture_hessians.py, ref_gptq_check.py, eval_ppl.py, gen_compare.py, summarize.py,
                 run_matrix.ps1
runs/            RESULTS-tinyllama.md = protocollo, tabella completa, testo generato, comandi di riproduzione
```
