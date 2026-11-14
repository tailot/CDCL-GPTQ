# CDCL-GPTQ su un modello medio: Gemma-2-2b-it

Misura del 2026-11-14 sulla stessa macchina di `RESULTS-tinyllama.md` (Intel Core Ultra 9 285K, 24 core; RTX 5070 Ti usata solo per catturare le Hessiane e per valutare). Stesso binario: `target\release\cdclq.exe`, `cdclq 1.0.0`, build release (`opt-level=3`, `lto="fat"`, `codegen-units=1`).

Questa è la seconda verifica su un LLM reale, e serve a vedere come si comporta il riparatore CDCL quando i layer hanno `d_in` grandi (9216 nel MLP) e l'architettura ha softcapping + GQA con 4 kv head.

## Protocollo

| | |
|---|---|
| modello | `Efficient-Large-Model/gemma-2-2b-it` — 2614 M parametri, Gemma-2: 26 layer, hidden 2304, MLP 9216, head dim 256 (8 q head → 2048, 4 kv head → 1024), `attn_logit_softcapping` 50, `final_logit_softcapping` 30. **Altup** (`per_layer_parameters`, 26 × 2304 = 59 904 pesi) non è implementato da transformers 4.52.4 e resta fp16 insieme ad embedding/lm_head/norm |
| pesi di partenza | il repo distribuisce due shard **bf16** (4990 + 241 MB = 5441 MB); fusi e convertiti in fp16 safetensors unico: **5229 MB** (`models/gemma-2-2b-it-fp16/model.safetensors`), per stare sullo stesso piano di partenza della griglia TinyLlama |
| calibrazione | WikiText-2 **train**, 128 sequenze × 2048 token = 262 144 token (`scripts/capture_hessians.py`, GPU) |
| Hessiane | due file, per non sforare i 16 GB di VRAM: `hess-gemma-attnmlp.safetensors` (78 matrici + 78 alias, q/k/v e gate/up condivisi) + `hess-gemma-down.safetensors` (26 matrici di `down_proj`, `d_in`=9216) = **10,4 GB** |
| layer quantizzati | 182 tensori 2-D = **2024 M parametri**; restano fp16 `embed_tokens` e `lm_head` (590 M), le tre norm per layer, i pesi altup e i softcap. Aliasse: q/k/v condividono l'Hessiana (78 matrici + 78 alias nei metadata) e cosi' pure gate/up |
| valutazione | WikiText-2 **test**, 285 chunk non sovrapposti da 1024 = 291 555 token contati (`scripts/eval_ppl.py`): perplexity + accuratezza top-1 next-token |
| confronto | GPTQ e CDCL con impostazioni identiche (act-order, damp 0.01, group 128, `--mse-grid`), pesi fake-quant caricati al posto degli originali |

Nota di ambiente: dal run TinyLlama del 13/11 l'ambiente python è cambiato (ora 3.14 + torch 2.14.1+cu130). Su Windows Triton non c'è, e transformers 4.52 invoca `torch.compile` dentro `generate()`: per le valutazioni serve `TORCHINDUCTOR_DISABLE=1`, altrimenti `eval_ppl.py` muore con `TritonMissing`. Nessun file del progetto è stato modificato.

## Risultati

| configurazione | bpw effettivo | modello (MB) | vs bf16 originale | tempo quant (s) | gain loss proxy | perplexity | Δppl vs fp16 | top-1 % | rel. L2 pesi |
|---|---|---|---|---|---|---|---|---|---|
| **fp16 (riferimento)** | 16.0 | 5229 | 1.04x | — | — | **17.875** | — | 45.44 | — |
| 4 bit, g128, mse-grid, cdcl | 4.19 | 2242 | 2.43x | 512 | −5.68% | **18.155** | +1.6% | 45.13 | 0.120 |
| 3 bit, g128, mse-grid, gptq | 3.19 | 1989 | 2.74x | 173 | — | 20.053 | +12.2% | 43.95 | 0.233 |
| 3 bit, g128, mse-grid, cdcl | 3.19 | 1989 | 2.74x | 640 | −6.84% | **19.926** | +11.5% | 43.94 | 0.234 |

Il riferimento bf16 del repo (5441 MB) sullo stesso testo fa ppl ~17,9 come la copia fp16: la conversione di dtype non cambia la valutazione. Il rapporto **vs bf16** è calcolato sul file `packed` completo, quindi include embedding + lm_head + norm rimasti in fp16 (590 M parametri ≈ 1180 MB): sono il collo di bottiglia del numero finale. Sui soli pesi quantizzati il guadagno è 16/3.19 = **5,0×** a 3 bit e 16/4.19 = **3,8×** a 4 bit (il file `dequant` è sempre 5229 MB, identico all'originale).

Lettura:

| **3 bit:** CDCL porta la loss proxy del **6,84%** (1826,68 → 1701,66) e la perplexity dello **0,63%** rispetto a GPTQ a parità di tutto (20.053 → 19.926). Su TinyLlama lo stesso identico set di impostazioni valeva il **16,6%**: qui il vantaggio si restringe molto.
- **4 bit:** quasi nessun beneficio (ppl +1,6% rispetto a fp16, modello già praticamente intatto), come atteso: a 4 bit GPTQ è vicino al pavimento.
- **Costo:** 3,7× il tempo di GPTQ (640 s vs 173 s per 2024 M di pesi su 24 core). Di questi, 63,6 s sono di sola preparazione (la Cholesky upper di H⁻¹) e **2,1 s per ogni `down_proj`**, perché lì `d_in`=9216: la fase di preparazione scala cubico con la dimensione di input.
- **Dove lavora il riparatore** (gain di loss proxy per tipo di layer, run 3 bit): `up_proj` +7,96%, `gate_proj` +7,34%, `q_proj` +6,37%, `v_proj` +6,13%, `k_proj` +5,84%, `o_proj` +4,10%, `down_proj` **+3,21%**. Il pattern è netto: il guadagno CDCL sale quando l'Hessiana è ben condizionata e scende dove è più malcondizionata. Su Gemma i 552 M parametri di `down_proj` (il 27% dei pesi quantizzati) sono proprio quelli che traggono meno vantaggio, e sono anche i più costosi da preparare: ecco perché il beneficio medio si ferma a −6,84%.

## Il modello funziona ancora?

Sì a 3 bit, ma con un calo qualitativo più visibile di TinyLlama (generazione greedy-side-by-side con `scripts/gen_compare.py`, seed fisso, temperature 0.8, top-p 0.95; output completo in `runs/gemma-gen.txt`):

| prompt | fp16 | 3 bit mse cdcl (2.74× più piccolo) |
|---|---|---|
| "The capital of France is" | "…known for its exquisite architecture, vibrant nightlife, and delicious food…" | "…" (si ferma subito: nessun token generato) |
| "The three main causes of the French Revolution were" | "…the social structure of the French society, the economic crisis, and the political system." | "…the Enlightenment, the American Revolution, and the social inequalities that existed in France." (corretto, anzi più preciso) |
| "Il giorno dopo andammo al mercato e" | "…ci fermammo davanti a un uomo che vendeva fiori…" | "…ci fermarono." (frase troncata) |
| "def fibonacci(n):\n    if n <= 1:" | si ferma alla prima riga | idem |

Cioè: la media (ppl +11,5%) non dice tutto. Il modello a 3 bit risponde ancora bene su una domanda di conoscenza, ma tende ad **anticipare la fine del periodo** (EOS prematuro) proprio dove fp16 continuava. È il modo in cui un modello da 2,6 B con embedding non quantizzato paga i 3 bit: top-1 perde solo 1,5 punti, ma la lunghezza generata crolla.

Verifica di integrità del formato compatto: `cdclq dequant` su `runs/gemma-b3mse-cdcl-fp16/packed/model.safetensors` restituisce **gli stessi 288 tensori, stessi dtype e forme, differenza massima 0.0** rispetto a `runs/gemma-b3mse-cdcl-fp16/dequant/model.safetensors`. Quindi il `packed` non perde nulla rispetto al modello fake-quant valutato qui sopra (e i numeri ppl/top-1 di `gemma-b3mse-gptq` e `gemma-b4mse-cdcl`, i cui dequant erano stati cancellati per spazio, sono stati ricalcolati ricostruendo i pesi in memoria dal `packed`).

## Cosa non va ancora bene (dal run Gemma)

- **Hessiane a pezzi obbligatorio:** 10,4 GB di Hessiane fp32 non stanno in 16 GB di VRAM insieme al modello. Servono due passate (`--skip down_proj` e poi `--only down_proj`) e `HessianStore` le unisce correttamente (78 + 26 matrici, alias preservati). Per un 7 B il numero di file cresce ancora.
- Il rapporto complessivo è frenato da embedding + lm_head in fp16: **590 M parametri**, il 23% del modello. Per scendere sotto ~2,7× servirebbe quantizzarli (basta passarli nelle Hessiane con `--skip ""` per lm_head); a parità di pesi quantizzati si salirebbe a ~4,6× sul file completo.
- `down_proj` è il collo di bottiglia dei tempi (26 layer × 2,1 s di sola preparazione) e il punto dove CDCL guadagna meno: un budget di bit non uniforme per tipo di layer qui sarebbe più utile della riparazione.
- Il vantaggio di CDCL sul ppl dipende molto dal modello: −16,6% su TinyLlama, −0,63% su Gemma a parità di impostazioni. Con i default attuali (group 128, mse-grid) non è garantito che la riparazione convenga sempre rispetto al costo di 3,7× in tempo.

## Come riprodurre

```powershell
# 0) ambiente (nessun file sorgente toccato)
& "$env:USERPROFILE\.cargo\bin\cargo.exe" build --release
#    transformers 4.52 + huggingface-hub < 1.0 nel virtualenv; dal 14/11 serve anche:
$env:TORCHINDUCTOR_DISABLE = "1"      # su Windows non c'e' Triton, generate() altrimenti esplode

# 1) modello: il repo e' aperto (gated=false), ma i pesi sono bf16 su due shard -> fp16 fuso
.venv\Scripts\python.exe runs\gemma-fp16.py        # fonde i 2 shard e scrive model.safetensors fp16

# 2) Hessiane (GPU, due passate per non sforare i 16 GB di VRAM)
.venv\Scripts\python.exe scripts\capture_hessians.py --model models\gemma-2-2b-it `
   --text data\calib.txt --out hess-gemma-attnmlp.safetensors --nsamples 128 --seqlen 2048 --batch 2 --skip down_proj
.venv\Scripts\python.exe scripts\capture_hessians.py --model models\gemma-2-2b-it `
   --text data\calib.txt --out hess-gemma-down.safetensors   --nsamples 128 --seqlen 2048 --batch 2 --only down_proj

# 3) quantizzazione (3 bit mse cdcl = configurazione principale)
.\target\release\cdclq.exe quantize --weights models\gemma-2-2b-it-fp16\model.safetensors `
   --hessians hess-gemma-attnmlp.safetensors --hessians hess-gemma-down.safetensors `
   --out-dir runs\gemma-b3mse-cdcl-fp16 --bits 3 --group 128 --mse-grid
#    variante GPTQ: aggiungere --method gptq ; variante 4 bit: --bits 4

# 4) valutazione ppl + top-1 (passare il FILE dequant, non la directory: eval_ppl.py ha un bug
#    nel branch "directory", variabile f riciclata nella list comprehension)
.\.venv\Scripts\python.exe scripts\eval_ppl.py --model models\gemma-2-2b-it-fp16 `
   --text data\wikitext2-test.txt --dequant runs\gemma-b3mse-cdcl-fp16\dequant\model.safetensors `
   --json runs\gemma-b3mse-cdcl-fp16.eval.json
#    per i run senza dequant su disco: .\.venv\Scripts\python.exe runs\eval_from_packed.py <nome-run>

# 5) testo generato (serve il wrapper per disattivare torch.compile di transformers)
.\.venv\Scripts\python.exe runs\gemma_gen_nocompile.py      # -> runs\gemma-gen.txt

# 6) verifica bit-esatta packed -> dequant
.\target\release\cdclq.exe dequant --input runs\gemma-b3mse-cdcl-fp16\packed\model.safetensors `
   --output runs\gemma-b3mse-cdcl-fp16\roundtrip.safetensors
```

File prodotti (enormi, da non versionare): `runs/<config>/{report.json,packed,dequant}`, `runs/<config>.eval.json`, `runs/<config>.log`, `hess-gemma-attnmlp.safetensors` (1,54 GB), `hess-gemma-down.safetensors` (8,83 GB), `models/gemma-2-2b-it-fp16/model.safetensors` (5,2 GB). I `dequant` Gemma sono 5,2 GB l'uno e i `roundtrip` altri 5,2 GB: servono solo alla valutazione e alla verifica di integrità, e sono stati cancellati per non saturare il disco (restano `packed` + `report.json` + gli JSON di valutazione).
