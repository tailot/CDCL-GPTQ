# cdclq — CDCL-GPTQ (Modulo B, corretto e implementato in Rust)

Quantizzatore *post-training* dei pesi per LLM (safetensors → INT2–INT8 a gruppi), scritto in Rust.
Parte da GPTQ (Cholesky esatto, aggiornamenti lazy a blocchi) e poi **ripara i conflitti**: usa la
decomposizione esatta della loss per trovare i sotto-blocchi peggiori e li migliora con due mosse
esatte e monotone (discesa per coordinate sui codici interi + raffinamento in forma chiusa delle
scale), alternandole fino a convergenza.

**Risultato misurato (benchmark sintetico, stessi bit, stessa griglia, stesso ordinamento, stessa
dampatura):** errore su token *held-out* ridotto del **−6% … −17%** rispetto a GPTQ (−9.6% a 3 bit,
−17.3% a 2 bit, −6.9% a 4 bit), al costo di **~1.5–5× il tempo** di GPTQ. Vedi "Risultati" e, subito
sotto, "Limiti": **non ho potuto misurare la perplexity su un LLM reale** (nessun modello, nessuna GPU
e nessun accesso a Hugging Face in questo ambiente).

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
```

Opzioni principali (`cdclq quantize --help`): `--bits`, `--group` (0 = una scala per riga),
`--damp`, `--no-act-order`, `--mse-grid`, `--skeleton-pct/--skeleton-bits`,
`--method gptq|cdcl` (per confronto), `--passes`, `--tol`, `--refine-frac`, `--no-refit-scales`,
`--only/--skip` (sottostringhe), `--threads`. Con più shard: `--weights a.safetensors b.safetensors`
(l'output mantiene i nomi dei file). Le Hessiane sono cercate come `<nome tensore senza .weight>.H`
(con alias da metadato `aliases`).

Profili di velocità: `--passes 2 --refine-frac 0.5` (veloce), `--passes 4` (bilanciato), default (completo).

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

- **Nessuna misura su LLM reali.** I numeri sopra vengono da layer sintetici. I guadagni dipendono dal
  condizionamento dell'Hessiana e dal numero di token di calibrazione; sui modelli reali possono essere
  minori o maggiori. Prima di fidarti: `capture_hessians.py` → `quantize` → perplexity di `out/dequant`
  contro la stessa quantizzazione con `--method gptq`.
- `scripts/capture_hessians.py` **non è stato eseguito** qui (serve torch + transformers); la sintassi è
  verificata, la logica no. Cattura le Hessiane del modello FP (calibrazione simmetrica); non produce `M`.
- Il motore supporta `M`, ma una pipeline *sequenziale* (input quantizzati layer dopo layer) richiede un
  driver che esegua il modello quantizzato: non incluso.
- Il formato `packed` usa permutazione act-order e zero a 8 bit (24 bit di metadati per gruppo e riga): **non
  ci sono kernel di inferenza**. Per valutare/usare il modello oggi serve `out/dequant` (pesi fake-quant).
- Solo CPU. Tempo ≈ 1.5–5× GPTQ ai default. Picco di memoria per layer ≈ 24·d_in² byte (d_in = 4096 → ~0.4 GB; 11008 → ~2.9 GB).
- Il Modulo A (`lllayer`) non è toccato da questo progetto.

## Struttura

```
src/linalg.rs    GEMM parallelo, Cholesky a blocchi, inversa triangolare, U di H⁻¹
src/quant.rs     griglie, scale f16, bit-packing, QLayer
src/engine.rs    GPTQ + conflitti + riparazione + target asimmetrico (+ test)
src/pipeline.rs  shard, alias, report, packed → dequant
src/io.rs        safetensors (F32/F16/BF16), Hessian store
src/bench.rs     benchmark sintetico
scripts/         capture_hessians.py, ref_gptq_check.py
```
