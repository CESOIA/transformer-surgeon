# HANDOFF.md — ripartire da contesto vuoto (refactoring 2026-10-28)

Questo file spiega a un agente (o a una persona) **dove si trova il refactoring** del
repository `transformer-surgeon` e **come proseguirlo**. Le cose ancora da fare sono in
[TODO.md](TODO.md). Leggere entrambi prima di toccare il codice.

## 1. Richiesta originale dell'utente (riassunto fedele)

Refactoring completo del repo mantenendo la struttura delle cartelle, pulendo e
semplificando il codice:

1. sincronizzare `draft-prolucio` con origin (fatto) e lavorare sul branch
   `draft-prolucio-refactor` (fatto, branch attuale);
2. esplorare il repo con un subagente per cartella e un riepilogo `.md` per cartella
   (fatto: `notes/refactor/recaps/*.md`);
3. eliminare il codice duplicato riducendolo a una sola funzione;
4. il codice obsoleto **non va cancellato ma archiviato** in `archive/`;
5. per ogni oggetto ci devono essere unit test in `test/`; i vecchi test vanno archiviati
   (fatto: `archive/tests/`), non cancellati;
6. documentazione leggibile in formato **sito web** dentro `docs/`, con esempi d'uso;
7. un **report ben scritto in italiano**, stile manuale tecnico: niente gergo né
   abbreviazioni inventate, ogni termine definito alla prima occorrenza; alla fine
   rileggerlo e chiarire le parti poco chiare.

### Regole di architettura (vincolanti)

- Il repo deve essere **generale**: pipeline unica compressione → export per quanti più
  transformer possibile.
- `utils/`: solo funzioni di gestione dei layer; **nessun** riferimento a modelli
  specifici (qwen, llama…), a tipi di compressione o all'export.
- `models/`: tutto ciò che distingue un modello, **in formato dizionario** come i file di
  indexing; è `models/` che dice come costruire il modello di export da `blocks/`.
- `compression/`: elabora i dizionari di compressione e applica la compressione.
- `calibration/`: solo raccolta di informazioni per le tecniche di compressione.
- `blocks/`: reimplementazione semplificata del transformer, divisa in sottocartelle di
  "atomi" selezionabili (una struttura per file) + un dizionario leggibile per sceglierli;
  nessuna informazione model-dependent.
- `export/`: esportatore generico che chiama esportatori per backend + un dizionario
  leggibile, usato dal codice, che collega tipo di blocco → operazione su dispositivo per
  ogni backend, con una opzione di default e le altre possibili (la scelta da parte
  dell'utente verrà progettata dopo).
- Obiettivo: aggiungere un modello tocca solo `models/`, un metodo di compressione solo
  `compression/`, una tecnica di calibrazione solo `calibration/` (salvo nuovi atomi in
  `blocks/` quando servono).

## 2. Stato del branch

Commit del refactoring (dal più vecchio):

| commit | contenuto |
|---|---|
| `046bfb7` | riepiloghi per cartella in `notes/refactor/recaps/` |
| `34fcaa4` | fase 1: `utils/` ridotto; manager/scheme/grouping/cascade/config → `compression/`; `convert_for_export` → `export/convert.py`; progetti collaterali in `archive/side_projects/` |
| `8e97dc3` | fasi 2–5: `blocks/` ad atomi + catalogo, `export/` stratificato + dizionario operazioni, `models/` a dizionari + fabbrica, pulizie in `compression/` e `calibration/` |
| `0a79938`, `b527fbd`, `2746d95` | esempio 03 corretto; nuova suite parziale; HANDOFF/TODO |
| fase 6 | nuova suite completa (`test/export/test_wrapper_manifest.py`, `test_exporter.py`, `test/hf/`), parità encoder, `pytest.ini` → `testpaths = test`, fixture `out_dir`/`torch_device` riesportate in `archive/tests/conftest.py` |

Il branch **non è ancora stato pushato** dopo il refactoring (solo `draft-prolucio` è su origin).

## 3. Nuova struttura del pacchetto (`transformersurgeon/`)

```
utils/        modules.py (get/set submodule, replace_parameter), indexing.py (flatten_index_paths,
              get_block_config, num_blocks…), devices.py, batches.py (normalize_batch, pack_batch), progress.py
blocks/       catalog.py  ← BLOCK_CATALOG: famiglia → atomo → {object, summary, implements, selected_by}
              layers/     LinearCompressed, EmbeddingCompressed, Conv2d/3dCompressed, VCONBlock, AtomicSum, LowRankMixin
              norm/       rmsnorm.py, rmsnorm_prescaled.py, rmsnorm_upcast.py, layernorm.py
              mlp/        plain.py (MLP), gated.py (MLPGated), activations.py
              positions/  rope.py, frequencies_default.py, frequencies_band_scaled.py ("llama3" = nome HF), rope_pruning.py
              attention/  base.py (MHABase), causal.py (MHACausal = kernel + cache), bidirectional.py (MHAEncoder)
                          kernels/  manual, sdpa, custom_sdpa (ExecuTorch), paged (Edge-LLM)
                          kv_cache/ mutable, io_scatter, io_concat, io_inplace, io_paged
              stacks/     decoder.py, encoder.py, config.py (riceve layer_specs = geometria dei lineari)
compression/  manager.py, scheme.py, grouping.py, cascade.py, config.py, compressible_layers.py,
              registry.py, compressori (lrd, structured/unstructured/block pruning, quantization, smoothquant)
              structured_pruning_methods/pruning_dims.py (spostato da blocks)
calibration/  backbone.py, raw_data/, summaries/, input_absmax.py (collect_input_absmax, usato da SmoothQuant),
              datasets.py (finestre di token per la calibrazione; WikiText-2 con import differito di `datasets`)
models/       factory.py (build_family); per famiglia: indexing_<m>.py (dizionario modello) + family_<m>.py (dizionario famiglia)
export/       operations.py ← BACKEND_OPERATIONS (blocco → operazione per backend, default + opzioni)
              exporter.py (Exporter, register_exporter, get_exporter, export_to_backend), config.py, convert.py,
              wrapper.py, quant_info.py, pt2e.py, pt2e_linear_quantizer.py, checks.py, manifest.py, result.py
              backends/  executorch_base.py, xnnpack.py, qnn/{exporter,passes}.py, onnx/, tensorrt/
hf/           hf_export.py (auto_map ora da `auto_class` del dizionario di famiglia)
```

Idee chiave:
- **Model → atomi**: il dizionario del modello indica le operazioni necessarie
  (`structure`, `attn_type`, `mlp_type`, `norm_type`, `rope_type`).
- **Export → implementazioni**: `export/operations.py` dà, per backend, l'implementazione di
  default (`cache_impl`, `attn_impl`, `rmsnorm_impl`, `add_batch_dim`, `masked_softmax`, e i
  campi config `quant_path`/`quant_scope`/`int4_backend`/`convert_linear_to_conv2d`).
  `resolve_convert_options(backend, options)` completa e valida; `check_components` valida
  un decoder già convertito. Alias accettati: `use_sdpa`, `rmsnorm_prescale`, `rmsnorm_upcast`.
- Default invariati per XNNPACK/QNN (`mutable`, `manual`, `rmsnorm_prescaled`); per
  ONNX/TensorRT il default è ora `io_inplace` (prima `mutable` dava errore).
- I blocchi non leggono più `compression_config`: `convert_for_export` legge dimensioni,
  rango e bias dai layer sorgente (`layer_specs`). La chiave `bias_required` è stata tolta
  dai dizionari dei modelli.
- I nomi di classe usati dall'export QNN (`MHACausal`, `TransformerDecoder`) sono rimasti.

## 4. Codice archiviato (`archive/`)

- `archive/side_projects/` — `qwen-vl-finetune/`, `vlm-modules-importance/` (non usati dalla libreria).
- `archive/code/` — funzioni morte o rotte, ognuna con docstring che spiega il motivo:
  M-RoPE helpers, `MHAEncoderFusedProj`, rami posizioni/patchify dell'encoder,
  `_apply_torchao_hard_quantization`, pezzi inutilizzati di `export/common.py`.
- `archive/scripts/` — script sostituiti (`qnn_full_quant_export.py` → `exporter_function_test.py --quant-layers all`).
- `archive/tests/` — vecchia suite completa, aggiornata ai nuovi import ed eseguibile:
  `pytest archive/tests` (riferimento di regressione finché non si decide di eliminarla).
- Rimossi senza archivio (non erano codice): `__init__.py` alla radice (commento),
  cartelle `test/*` contenenti solo `__pycache__`, metodi vuoti/inutilizzati del manager
  (`update_config`, `_set_model`, attributo `smoothing`), chiavi mai lette dei dizionari
  (`qkv_paths`, `pruning_supported`, `lrd_supported`, `per_head_uniform`).

## 5. Correzioni di comportamento introdotte (da citare nel report)

- `LinearCompressed` completamente potato (`out_features=0`) non inizializzava `nn.Module`
  e andava in errore se chiamato: corretto.
- Encoder convertito: ora usa `layer_norm_eps` del modello (prima sempre 1e-5; BERT usa 1e-12).
  Verificato: BERT, DistilBERT, ViT convertiti coincidono con HF (scarto ≤ 2e-5).
- Qwen2.5-VL: la classe modello non dichiarava `config_class` (annotazione al posto di
  assegnazione); la fabbrica ora lo imposta.
- `models/__init__.py` non inghiotte più ogni eccezione: solo `ImportError`, con warning.
- Il dizionario di export garantisce la coppia cache `io_paged` ↔ kernel `paged`
  (`KVCache.requires_kernel`).
- `calibration_token_ids` disponibile anche per XNNPACK.

## 6. Ambiente e comandi

- Repo: `/home/AD/luciano.prono/container-py310-executorch-qnn/transformer-surgeon`.
- Usare **solo gli ambienti conda** (il `python` di base importa un'altra copia del repo):
  `py312_tensorrt` (veloce, ha onnx/tensorrt/GPU, ha `pyflakes`), `py310-executorch-qnn`,
  `py312_executorch_xnnpack` (pytest installato durante questa sessione).
- Gli script fuori dal repo vanno lanciati con `PYTHONPATH=$PWD`. Attenzione: `conda run` **sovrascrive**
  `PYTHONPATH`; usare `conda run -n <env> env PYTHONPATH=<repo> python ...`.
- Test (`pytest.ini` raccoglie solo `test/`):
  ```bash
  conda run -n py312_tensorrt python -m pytest test -q -p no:cacheprovider          # nuova suite
  conda run -n py312_tensorrt python -m pytest archive/tests -q -p no:cacheprovider --continue-on-collection-errors # vecchia suite
  conda run -n py312_tensorrt python -m pyflakes transformersurgeon                  # nomi non definiti
  ```
- **Fallimenti attesi d'ambiente** (presenti già prima del refactoring, nella vecchia suite):
  - `py310-executorch-qnn`: `test_precision_passes` (manca `onnx`), `test_known_bugs::test_qnn_unavailable…`
    (l'SDK QNN è presente), 2 test `custom_sdpa` (operatori ExecuTorch mancanti),
    `test_export_xnnpack*` (out-variants quantizzati mancanti), `test_export_qnn` (IndexError nel toolchain);
  - `py312_executorch_xnnpack`: `test_precision_passes`, `test_export_xnnpack`, `test_export_xnnpack_custom_sdpa`;
  - `py312_tensorrt`: nessuno (145 passati, 18 saltati).
  Senza `--continue-on-collection-errors` la vecchia suite negli ambienti ExecuTorch si ferma
  all'errore di raccolta di `test_precision_passes` (manca `onnx`).
  **Nuova suite**: nessun fallimento atteso in nessun ambiente (200/4, 198/6, 198/6 passati/saltati).
  Nota: nel test QNN della nuova suite si usa `soc_model="SM8850"`, `quant_scope="all"` perché con lo
  scope di default questa versione del toolchain QNN non convalida alcuni operatori float sull'HTP
  (`aten.rms_norm` non delegato); il vecchio `test_export_qnn` fallisce invece con IndexError.
  Qualsiasi altro fallimento è una regressione.

## 7. Convenzioni di lavoro

- **Tenere HANDOFF.md e TODO.md sempre aggiornati** dopo ogni passo (richiesta dell'utente): devono
  bastare per ripartire da contesto vuoto. Il file si chiamava `AGENT.md` (rinominato).

- Commit piccoli per fase, messaggi in inglese che terminano con:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Xh2BkoWSaNL9unVPZitrQ6
  ```
- Dopo ogni modifica alla libreria: suite in `py312_tensorrt`, poi negli altri due ambienti,
  confrontando con l'elenco dei fallimenti attesi.
- Non lanciare test mentre si modificano moduli che i test importano in modo differito
  (gli esportatori dei backend).
- Il file `AGENTS.md` (con la S) e `CLAUDE.md` descrivono ancora la **vecchia** struttura:
  vanno aggiornati (vedi TODO).

## 8. Duplicazioni già eliminate (per il report)

| prima (copie) | ora (una sola definizione) |
|---|---|
| `init_lrd`/`cancel_lrd` in `LinearCompressed` e `EmbeddingCompressed` | `blocks/layers/low_rank.py::LowRankMixin` |
| costruzione dei lineari ripetuta in MHA, MLP, MLPGated (rank/bias da `compression_config`) | `blocks/layers/linear_compressed.py::linear_from_spec` |
| tre varianti RMSNorm in una classe con flag | un atomo per file + `rms_normalize` condivisa |
| scrittura cache + kernel con `if/elif` in `MHACausal.forward`, maschere nel decoder | atomi `kv_cache/*` e `kernels/*` con interfaccia comune |
| `from_source_config` decoder vs encoder; split per prefisso `blocks.{i}.` decoder vs encoder | `stacks/config.py::_ConvertedStackConfig` (+ `block_layer_specs`) |
| 8 file `define_*.py` quasi identici (config/modello/manager) | `models/factory.py::build_family` + dizionari di famiglia |
| `export/common.py` (1180 righe) | `wrapper.py`, `quant_info.py`, `pt2e.py`, `checks.py`, `manifest.py`, `result.py` |
| tre parser dell'input di export; normalizzazione dei device fatta due volte | `export/exporter.py::resolve_components` + `Exporter.export` |
| scrittura `.pte` + finalize ripetuti in XNNPACK (2×) e QNN | `ExecuTorchExporter.write_program` / `finalize` |
| lista dei 19 attributi di quantizzazione in `quantization.py` e `convert.py` | `QUANT_RUNTIME_BUFFERS` / `QUANT_RUNTIME_ATTRS` |
| `_extract_aqt_scale` copiata in due file | `compression/quantization.py::extract_torchao_scale` |
| hook di quantizzazione simulata ingresso/uscita; regola qmin/qmax ripetuta 3× | `_make_fake_quant_hook` + `activation_qrange` |
| `WeightCollector`/`BiasCollector`, `WeightSummary`/`BiasSummary`, due riepiloghi di intervallo | `ParamCollector`, `_ParamSummary`, `_RangeSummary` |
| dizionari `__ts_args__` costruiti a mano in 5 punti della cascata | `utils/batches.py::pack_batch` / `normalize_batch` |
| `config if config_attr == "" else getattr(...)` in 6 punti | `utils/indexing.py::get_block_config` / `num_blocks` |
| raccolta absmax di SmoothQuant fuori da `calibration/` | `calibration/input_absmax.py::collect_input_absmax` |
| caricatore WikiText-2 copiato in 6 script/esempi + 2 benchmark | `calibration/datasets.py::token_windows_loader` / `wikitext2_loader` |
| `scripts/executorch/qnn/full_quant_export.py` (~90% copia) | `exporter_function_test.py --quant-layers all` |
| `isinstance(compressor, Quantizer)` nel manager | hook generico `Compressor.prepare_for_save` |
| nome del componente "text" fisso nell'export | componente con `structure == "transformer_decoder"` |
