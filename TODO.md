# TODO — refactoring `draft-prolucio-refactor`

Leggere prima [HANDOFF.md](HANDOFF.md). Ordine consigliato; spuntare man mano.

## A. Nuova suite di test — FATTA (commit "Refactor phase 6")

Nuova suite `test/` verde nei tre ambienti (2026-10-28): `py312_tensorrt` 211 passati / 4 saltati,
`py312_executorch_xnnpack` 209/6, `py310-executorch-qnn` 209/6. Vecchia suite `archive/tests`:
solo i fallimenti d'ambiente attesi di HANDOFF.md §6 (nessuna regressione).
File: `test/{utils,blocks,compression,calibration,models}/`, `test/export/{test_operations,test_convert,
test_wrapper_manifest,test_exporter}.py`, `test/hf/test_hf_export.py`.

## B. Pulizie rimaste

- [x] Script: caricatore WikiText-2 copiato in 8 file → `calibration/datasets.py`
      (`token_windows_loader`, `wikitext2_loader`; verificato: stessi campioni del vecchio caricatore).
- [x] `scripts/executorch/qnn/full_quant_export.py` unito in `exporter_function_test.py`
      (`--quant-layers {none,mlp,all}`, `--quant-method`, `--act-method`, `--act-scheme`; alias
      `--quant-mlp`, `--precision`, `--quantize-activations`); originale in `archive/scripts/`;
      `run_full_quant.sh` = `exporter_function_test.py --quant-layers all`. Le citazioni di
      `full_quant_export.py` in `AGENTS.md` e `notes/refactor/recaps/` vanno sistemate al punto C.
- [x] Validatori duplicati in `compression/`: `compression/validators.py` (`one_of`, `in_range`,
      `positive_number`, `positive_int`, `boolean`); i nomi `validate_*` restano (registro invariato).
      Tenuti a mano solo `repeated_pattern`, `coupled_repeated_pattern`, `blocks` (messaggi esplicativi).
      Messaggi ora uniformi: `"<nome> must be <regola>, but got <valore>."`.
- [x] Nomi I/O del grafo: una sola definizione `export/manifest.py::cache_io_name` / `graph_io_names`
      (`onnx/exporter.py::io_names` ora è un alias).
- [x] `pyflakes transformersurgeon` pulito: restano solo le ri-esportazioni degli `__init__` e 5 import
      voluti (registrazione di operatori/esportatori: `custom_sdpa`, `exporter.py` → `backends`,
      `qnn/passes.py`, `onnx/exporter.py`, `tensorrt/edgellm_attention.py`).
- [x] `hf/hf_export.py` chiama ancora `manager.apply(hard=True)` se riceve un manager: **tenuto** per
      compatibilità e documentato nella docstring come unica eccezione; da citare nel report.

## B2. Violazioni residue delle regole di architettura (verificate il 2026-10-28)

Nessun ramo di codice specifico per un modello fuori da `models/`; restano:
- [x] Commenti che citavano modelli in `compression/`, `export/`, `blocks/` riformulati in modo generico
      (la caratteristica al posto del modello; le misure restano come "decoder da 1.5B parametri").
      Resta solo `llama::custom_sdpa` in `export/operations.py`, nome di un operatore ExecuTorch.
- [ ] `compression/structured_pruning.py` conosce la topologia (sentinelle `preprocessing`,
      `final_layer`, `final_norm`, `preprocessing_conv`, `_hop_block_id`): spostare la risoluzione dei
      percorsi in un'API dello schema/manager, così il compressore riceve catene di moduli già risolte.
- [x] `compression/block_pruning.py`: verificato che è già generico (blocchi di righe raggruppati in unità;
      i nomi degli attributi `owner`/`block_size_attr`/`blocks_per_unit_attr` vengono dal dizionario del
      modello). Rispetta la regola: da spiegare nel report. Corretti anche i riferimenti a file spostati
      nelle docstring (`blocks/mha.py`, `blocks/rope.py`, `utils/cascade.py`, `utils/modeling.py`, ...).
- [x] Contratto compressione→export: gli attributi privati (`_torchao_*`, `_act_quant_*`, ...) ora sono
      letti solo in `compression/quantization.py`, tramite le funzioni pubbliche `export_quant_info(module)`
      (descrittore usato da `export/quant_info.py`) e `copy_quant_state(old, new)` (usata da `export/convert.py`).
      Codice spostato senza cambi di logica.
- [x] Export QNN: l'attenzione si riconosce dal marcatore `EXPORT_ROLE` sugli atomi (`MHACausal` =
      "attention", `TransformerDecoder` = "decoder_stack"), letto da `qnn/passes.py::_export_role`;
      verificato: stessi nodi del vecchio confronto per nome (mutable, io_inplace, sdpa).
- [x] `hf/hf_export.py::export_to_hf` chiama `manager.apply(hard=True)`: tenuto e documentato (punto B).
- [ ] `blocks/` contiene atomi legati a una backend (`custom_sdpa` ExecuTorch, `paged` Edge-LLM):
      ammesso perché sono atomi selezionabili, ma va detto nel report.
- [x] `BACKEND_OPERATIONS` ora copre anche `mlp`, `rope_frequencies`, `embedding`, `output_head` con un
      terzo tipo di voce `model_key` (scelta del dizionario del modello, `default: None`, o implementazione
      unica): solo documentazione dell'operazione su dispositivo, il codice di risoluzione non cambia.
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
      cartella (prima/dopo); le duplicazioni eliminate (tabella in HANDOFF.md §8); i due dizionari (`BLOCK_CATALOG`, `BACKEND_OPERATIONS`) con esempi;
      cosa è stato archiviato e perché; correzioni di comportamento (HANDOFF.md §5); duplicazioni
      eliminate; test (vecchi archiviati, nuovi per oggetto, fallimenti d'ambiente attesi);
      come aggiungere un modello / un metodo di compressione / una calibrazione / un backend;
      limiti e lavoro futuro (es. selezione utente delle opzioni non-default, modelli con QKV
      fuso come ModernBERT non esportabili, Qwen-VL senza `structure`).

## F. Chiusura

- [ ] Suite completa nei tre ambienti, nessuna regressione rispetto ad HANDOFF.md §6.
- [ ] Ripassare la richiesta originale (HANDOFF.md §1) punto per punto e confermare che ogni regola è
      rispettata o che l'eccezione è motivata nel report.
- [ ] `git push -u origin draft-prolucio-refactor` (chiedere conferma all'utente).
