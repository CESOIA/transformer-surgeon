# TODO — refactoring `draft-prolucio-refactor`

Leggere prima [AGENT.md](AGENT.md). Ordine consigliato; spuntare man mano.

## A. Nuova suite di test (in corso)

Fatti e verdi in `py312_tensorrt` (non ancora eseguiti negli altri due ambienti):
`test/utils/` (21), `test/blocks/` (52 + 1 saltato), `test/compression/` (40),
`test/calibration/` (5), `test/models/` (33), `test/export/test_operations.py`,
`test/export/test_convert.py`.

- [ ] `test/export/test_convert.py`: aggiungere un test di **parità numerica degli encoder**
      (BERT, DistilBERT, ViT contro HF). Lo script pronto è in
      `/tmp/claude-10033318/.../scratchpad/encoder_parity.py` (potrebbe non esistere più):
      BERT → `m.bert.encoder(m.bert.embeddings(ids))`, DistilBERT → `m.distilbert.transformer(emb, attn_mask=ones)`,
      ViT → ciclo su `m.vit.layers` + `m.vit.layernorm`. Scarti misurati ≤ 2e-5.
- [ ] `test/export/test_wrapper_manifest.py`: `LLMWrapper` (firme per cache interna / I/O / pagine),
      `build_zero_caches`, `build_example_inputs`, `build_llm_manifest` + `load_llm_manifest` +
      `zero_caches_from_manifest`, `extract_layer_quant_info` dopo quantizzazione hard,
      `finalize_export_result` che scrive il manifest, `find_weight_mismatches`.
- [ ] `test/export/test_exporter.py`: registro (`EXPORTERS` con 4 backend, `get_exporter` su nome
      ignoto → errore, config di tipo sbagliato → `TypeError`), export ONNX reale su modello
      minuscolo (gated `requires_onnx`, vedi `archive/tests/unit/test_onnx_tensorrt_export.py`),
      TensorRT (gated GPU), XNNPACK e QNN (gated, `test/_helpers/capabilities.py`), pass QNN
      (`_split_float_linear_bias`, vedi `archive/tests/unit/test_qnn_bias_split.py`), pass TensorRT
      (`fp32_accumulate_lrd`, `fp32_attention_scores`).
- [ ] `test/hf/test_hf_export.py`: `export_to_hf` in locale (senza Hub): salvataggio, `auto_map`
      con `auto_class`, ricarica con risultati identici.
- [ ] Aggiornare `pytest.ini`: `testpaths = test` (oggi punta a `test/unit test/e2e`, ora archiviati).
- [ ] Eseguire la nuova suite nei tre ambienti; confrontare con `pytest archive/tests` (vecchia suite).
- [ ] Commit "Refactor phase 6: new test suite mirroring the package".

## B. Pulizie rimaste

- [ ] Script: il caricatore WikiText-2 di calibrazione è copiato in ~6 file
      (`scripts/executorch/*`, `benchmarks/nonhom_*`, `examples/02`): ridurlo a una funzione
      (es. `scripts/common_data.py` o, se generale, `calibration/datasets.py`).
- [ ] `scripts/executorch/qnn/full_quant_export.py` è ~90% copia di `exporter_function_test.py`:
      unificare o archiviare.
- [ ] Validatori duplicati in `compression/` (ratio, eps, method): un modulo `validators.py` con fabbriche.
- [ ] `export/backends/onnx/exporter.py::io_names` duplica i nomi I/O di `manifest.py`: unificare.
- [ ] `utils/__init__` va bene; verificare con `pyflakes` che non restino import inutili
      (restano solo import fatti apposta per registrare operatori).
- [ ] Valutare se `hf/hf_export.py` debba ancora chiamare `manager.apply(hard=True)` (compressione
      lanciata dall'export HF): documentarlo o spostarlo.

## B2. Violazioni residue delle regole di architettura (verificate il 2026-10-28)

Nessun ramo di codice specifico per un modello fuori da `models/`; restano:
- [ ] 27 **commenti** che citano modelli (Qwen, Llama, BERT, ViT) in `compression/`, `export/`,
      `blocks/` (`grep -rniE "qwen|llama|bert" transformersurgeon --include=*.py`, escludendo
      `torch.ops.llama`/`llama.fallback`/`llama3` che sono nomi di librerie): riformulare in modo generico.
- [ ] `compression/structured_pruning.py` conosce la topologia (sentinelle `preprocessing`,
      `final_layer`, `final_norm`, `preprocessing_conv`, `_hop_block_id`): spostare la risoluzione dei
      percorsi in un'API dello schema/manager, così il compressore riceve catene di moduli già risolte.
- [ ] `compression/block_pruning.py` è specifico per le teste di attenzione e modifica un attributo
      del modello HF (`owner.<blocks_per_unit_attr>`): documentare o generalizzare.
- [ ] Contratto compressione→export tramite attributi privati (`_torchao_*`, `_act_quant_*`,
      `_soft_quant_precision`) letti da `export/quant_info.py`: introdurre un descrittore pubblico
      (es. `Quantizer.export_info(module)`).
- [ ] Export QNN riconosce l'attenzione cercando i nomi di classe `MHACausal`/`TransformerDecoder`
      in `nn_module_stack` (`export/backends/qnn/passes.py::_float_attention_nodes`): usare un
      marcatore sull'atomo.
- [ ] `hf/hf_export.py::export_to_hf` chiama `manager.apply(hard=True)` (vedi punto B).
- [ ] `blocks/` contiene atomi legati a una backend (`custom_sdpa` ExecuTorch, `paged` Edge-LLM):
      ammesso perché sono atomi selezionabili, ma va detto nel report.
- [ ] `BACKEND_OPERATIONS` copre attenzione, cache, RMSNorm, batch, softmax mascherata e lineari;
      **mancano** MLP, RoPE, embedding e testa di uscita (oggi una sola implementazione: aggiungere
      voci con il solo default per completezza, oppure spiegarlo nel report).
- [ ] Verifica "un'aggiunta tocca una sola cartella": provare a scrivere (anche solo in bozza) una
      nuova famiglia di modello, un nuovo metodo LRD, un nuovo riepilogo di calibrazione e annotare
      quali file servono; riportarlo nel report.

## C. Documentazione di progetto

- [ ] Riscrivere `AGENTS.md` e `CLAUDE.md` (mappa file, invarianti, ricette "aggiungi modello /
      metodo / calibrazione / atomo / backend") sulla nuova struttura.
- [ ] Aggiornare `README.md`, `FRAMEWORK_STRUCTURE.md`, `llms.txt`, `docs/concepts.md`
      (es. `weight_2` → `linear_V`, percorsi `utils.manager` → `compression.manager`,
      `utils.convert` → `export.convert`, `export_with_*` → `export_to_backend`/classi `*Exporter`).

## D. Sito web della documentazione (`docs/`)

- [ ] `mkdocs`/`mkdocs-material`/`mkdocstrings` non sono installati in nessun ambiente:
      installarli in `py312_tensorrt` (chiedere conferma all'utente) oppure generare HTML statico.
- [ ] Pagine: introduzione e concetti; pipeline compressione → export; una pagina per cartella
      (`utils`, `blocks` con tabella di `BLOCK_CATALOG`, `compression`, `calibration`, `models`
      con lo schema del dizionario del modello e del dizionario di famiglia, `export` con la
      tabella di `BACKEND_OPERATIONS`); ricette "come aggiungere…"; esempi d'uso (01–04);
      riferimento API (mkdocstrings). Aggiornare `mkdocs.yml` (nav, `docs/api/*` con i nuovi percorsi).
- [ ] Le tabelle dei due dizionari conviene generarle da codice (script in `docs/`), così restano allineate.
- [ ] `docs/investigations/` resta come storico.

## E. Report finale (in italiano)

- [ ] File suggerito: `notes/refactor/REPORT.md` (e una pagina nel sito). Stile manuale tecnico:
      niente gergo o abbreviazioni inventate, ogni termine definito alla prima occorrenza
      (es. "decomposizione a basso rango (LRD, low-rank decomposition)", "cache KV",
      "quantizzazione simulata", "backend"). Rileggerlo alla fine e chiarire le parti oscure.
- [ ] Contenuti: obiettivo e regole; metodo (riepiloghi per cartella); cosa è cambiato per
      cartella (prima/dopo); le duplicazioni eliminate (tabella in AGENT.md §8); i due dizionari (`BLOCK_CATALOG`, `BACKEND_OPERATIONS`) con esempi;
      cosa è stato archiviato e perché; correzioni di comportamento (AGENT.md §5); duplicazioni
      eliminate; test (vecchi archiviati, nuovi per oggetto, fallimenti d'ambiente attesi);
      come aggiungere un modello / un metodo di compressione / una calibrazione / un backend;
      limiti e lavoro futuro (es. selezione utente delle opzioni non-default, modelli con QKV
      fuso come ModernBERT non esportabili, Qwen-VL senza `structure`).

## F. Chiusura

- [ ] Suite completa nei tre ambienti, nessuna regressione rispetto ad AGENT.md §6.
- [ ] Ripassare la richiesta originale (AGENT.md §1) punto per punto e confermare che ogni regola è
      rispettata o che l'eccezione è motivata nel report.
- [ ] `git push -u origin draft-prolucio-refactor` (chiedere conferma all'utente).
