# Rifattorizzazione di transformer-surgeon — relazione tecnica

Data: 28 ottobre 2026. Ramo: `draft-prolucio-refactor` (derivato da `draft-prolucio`).

Questa relazione descrive la riorganizzazione del codice della libreria `transformer-surgeon`: perché è stata
fatta, con quale metodo, che cosa è cambiato in ogni cartella, che cosa è stato archiviato, come è stata
verificata e quali limiti restano. È scritta per chi deve usare o modificare la libreria senza aver seguito il
lavoro. I nomi di file, funzioni e classi sono scritti in `carattere a spaziatura fissa` e si riferiscono al
pacchetto Python `transformersurgeon/`, salvo dove indicato.

---

## 1. Termini usati

I termini seguenti ricorrono in tutta la relazione. Sono definiti qui una volta per tutte.

- **Modello transformer.** Rete neurale composta da una sequenza di *blocchi* uguali, ognuno formato da un
  meccanismo di *attenzione* (ogni posizione della sequenza combina informazioni dalle altre posizioni) e da una
  rete *feed-forward* (due o tre trasformazioni lineari con una funzione di attivazione in mezzo, indicata nel
  codice come MLP, *multi-layer perceptron*). Esistono *decoder* (generano testo una parola alla volta e vedono
  solo il passato: Qwen2, Llama) ed *encoder* (leggono tutta la sequenza: BERT, ViT).
- **Layer, pesi, embedding.** Un *layer* (strato) è una singola operazione della rete con i suoi parametri; i più
  comuni sono i layer *lineari*, che moltiplicano l'ingresso per una matrice di *pesi* e aggiungono un vettore
  detto *bias*. L'*embedding* è il primo layer: trasforma ogni parola (rappresentata da un numero intero, il
  *token*) in un vettore. La *testa di uscita* è l'ultimo layer: trasforma il vettore finale in un punteggio per
  ogni parola del vocabolario.
- **Teste di attenzione.** L'attenzione calcola, per ogni posizione, tre vettori: *interrogazione* (in inglese
  *query*), *chiave* e *valore*. Il calcolo è diviso in più *teste* indipendenti, ognuna su una porzione del
  vettore. Nell'attenzione a gruppi di interrogazioni (*grouped-query attention*) più teste di interrogazione
  condividono la stessa testa di chiave e valore.
- **Precisione numerica.** I numeri reali sono rappresentati in *virgola mobile* a 32 bit (*precisione
  singola*) o a 16 bit (*mezza precisione*, più veloce ma con un intervallo di valori molto più piccolo).
- **Batch, maschera, kernel.** Un *batch* è un gruppo di esempi elaborati insieme. Una *maschera* è un vettore di
  zeri e uni che indica quali elementi tenere (nella potatura) o quali posizioni ignorare (nell'attenzione). Un
  *kernel* è l'implementazione di un'operazione per un certo hardware.
- **Hugging Face.** Libreria di riferimento (`transformers`) da cui provengono i modelli. La libreria
  `transformer-surgeon` ne estende le classi senza cambiarne l'uso (`from_pretrained`, `generate`,
  `save_pretrained`).
- **Compressione.** Insieme di tecniche che riducono la dimensione o il costo di calcolo di un modello. La
  libreria ne offre quattro:
  - **decomposizione a basso rango** (in inglese *low-rank decomposition*, nel codice `lrd`): una matrice dei
    pesi di dimensione `uscite × ingressi` viene sostituita dal prodotto di due matrici più piccole,
    `uscite × r` e `r × ingressi`, dove `r` (il *rango*) è minore di entrambe le dimensioni;
  - **potatura strutturata** (*structured pruning*): si eliminano righe intere di una matrice, cioè neuroni
    di uscita; i layer che ricevono quelle uscite perdono le colonne corrispondenti (*potatura accoppiata*);
  - **potatura a blocchi** (*block pruning*): si eliminano gruppi interi di righe, ad esempio intere teste di
    attenzione;
  - **potatura non strutturata** (*unstructured pruning*): si azzerano singoli pesi senza cambiare le forme;
  - **quantizzazione**: i pesi (ed eventualmente gli ingressi, detti *attivazioni*) vengono rappresentati con
    interi di pochi bit (ad esempio 8 o 4) e un fattore di scala, invece che con numeri in virgola mobile.
- **Applicazione "soft" e "hard".** Nella forma *soft* la compressione è simulata e reversibile (ad esempio i
  pesi quantizzati vengono riconvertiti subito in virgola mobile, le righe potate vengono azzerate). Nella forma
  *hard* è definitiva (le righe vengono tolte, i pesi diventano interi).
- **Calibrazione.** Passaggio di dati reali nel modello per raccogliere statistiche (ad esempio la covarianza
  degli ingressi di un layer, o il loro intervallo di valori) che alcune tecniche di compressione usano per
  scegliere che cosa eliminare o quale scala usare.
- **Export.** Trasformazione del modello compresso in un file eseguibile su un dispositivo specifico.
- **Backend.** Il software di esecuzione del dispositivo di destinazione. La libreria ne supporta quattro:
  **XNNPACK** (processori ARM e x86, tramite il formato `.pte` di ExecuTorch, il runtime mobile di PyTorch),
  **QNN** (unità neurali dei processori Qualcomm, sempre tramite ExecuTorch), **ONNX** (formato portabile) e
  **TensorRT** (schede grafiche NVIDIA, costruito a partire dal file ONNX).
- **Cache KV.** Durante la generazione, un decoder conserva le *chiavi* e i *valori* dell'attenzione delle parole
  già elaborate, per non ricalcolarli. Questa memoria si chiama cache delle chiavi e dei valori (cache KV).
- **Grafo di export.** Versione semplificata del modello, scritta con operazioni che i backend sanno tradurre,
  che viene *tracciata* (registrata operazione per operazione) da `torch.export`.
- **Gestore, schema, compressore.** Il *gestore* (`CompressionSchemesManager`) è l'oggetto con cui l'utente
  configura e applica la compressione a tutto il modello. Per ogni layer comprimibile esiste uno *schema*
  (`CompressionScheme`) che conserva le opzioni di quel layer. Un *compressore* (ad esempio `LRDer`,
  `Quantizer`) è l'algoritmo che comprime un singolo layer.
- **Termini di sviluppo.** Una *docstring* è il testo di documentazione scritto all'inizio di una funzione o
  classe Python. Un *commit* è un salvataggio registrato nella storia del codice (strumento `git`), un *ramo*
  (*branch*) è una linea di sviluppo separata. Un *SDK* (*software development kit*) è il pacchetto di strumenti
  di un produttore; un *plugin* è un componente aggiuntivo che estende un programma con nuove operazioni.
- **Dizionario.** In questa relazione, una struttura dati Python (`dict`) leggibile che descrive qualcosa al
  posto di codice: un modello, un catalogo di componenti, una tabella di scelte. Il codice legge i dizionari
  invece di contenere rami del tipo "se il modello è X".

---

## 2. Obiettivo e regole

### 2.1 La richiesta

Rifattorizzare l'intero repository mantenendo la struttura delle cartelle, pulendo e semplificando il codice:

1. eliminare il codice duplicato, riducendo ogni duplicazione a una sola funzione;
2. non cancellare il codice obsoleto, ma archiviarlo nella cartella `archive/`;
3. fornire test unitari per ogni oggetto nella cartella `test/`, archiviando (non cancellando) i test vecchi;
4. scrivere la documentazione in forma di sito web nella cartella `docs/`, con esempi d'uso;
5. scrivere questa relazione.

### 2.2 Le regole di architettura

La libreria deve offrire **una sola catena compressione → export**, valida per il maggior numero possibile di
modelli transformer. Ogni cartella ha un compito preciso:

| Cartella | Compito | Non deve contenere |
|---|---|---|
| `utils/` | funzioni generiche di gestione dei layer (ricerca e sostituzione di sottomoduli, percorsi) | riferimenti a modelli, a tecniche di compressione o all'export |
| `models/` | tutto ciò che distingue un modello, **in forma di dizionario**; indica come costruire il modello di export a partire da `blocks/` | algoritmi |
| `compression/` | elaborazione dei dizionari di compressione e applicazione della compressione | riferimenti a modelli specifici, dettagli di export |
| `calibration/` | solo raccolta di informazioni per le tecniche di compressione | decisioni di compressione |
| `blocks/` | reimplementazione semplificata del transformer, divisa in *atomi* selezionabili (un componente per file) e un dizionario leggibile che li elenca | informazioni dipendenti dal modello |
| `export/` | un esportatore generico che chiama gli esportatori dei singoli backend, e un dizionario leggibile che collega ogni tipo di blocco all'operazione eseguita sul dispositivo, per ogni backend, con un'opzione predefinita e le alternative | decisioni di compressione |

La conseguenza voluta è che **aggiungere un modello tocchi solo `models/`**, un metodo di compressione solo
`compression/`, una tecnica di calibrazione solo `calibration/` e un backend solo `export/` (con l'eccezione di
nuovi atomi in `blocks/`, quando un'operazione non esiste ancora).

---

## 3. Metodo

1. **Allineamento.** Il ramo `draft-prolucio` è stato sincronizzato con il repository remoto e il lavoro è stato
   svolto su un nuovo ramo, `draft-prolucio-refactor`.
2. **Esplorazione.** Ogni cartella è stata esaminata separatamente e riassunta in un file in
   `notes/refactor/recaps/` (una pagina per cartella: contenuto, dipendenze, duplicazioni, codice morto, violazioni
   delle regole). Questi riassunti sono la base delle decisioni descritte sotto.
3. **Fasi.** Il lavoro è stato diviso in fasi, ognuna chiusa da un commit e da una verifica completa dei test in
   tre ambienti (sezione 9): prima `utils/`, poi `blocks/`, `export/`, `models/`, `compression/` e
   `calibration/`, poi la nuova suite di test, infine le pulizie residue, la documentazione e questa relazione.
4. **Regola di non regressione.** Ogni spostamento di codice è stato fatto senza cambiare la logica; quando un
   comportamento è cambiato (sezione 7) è stato per correggere un errore, ed è documentato. Dove possibile, il
   comportamento nuovo è stato confrontato con quello vecchio sugli stessi ingressi (ad esempio: stessi nodi
   selezionati, stessi campioni di calibrazione, stesse uscite numeriche).

In cifre: il pacchetto passa da 124 file e 17 838 righe a 165 file e 17 073 righe. I file sono di più perché molte
classi che convivevano in un unico file sono ora in un file ciascuna; le righe sono di meno perché le copie sono
state eliminate.

---

## 4. Che cosa è cambiato, cartella per cartella

### 4.1 `utils/`

**Prima.** La cartella conteneva il gestore della compressione, gli schemi, i gruppi, la configurazione, la
sostituzione dei layer e la conversione per l'export: cioè quasi tutta la logica della libreria.

**Dopo.** Restano solo funzioni generiche:

- `modules.py`: ricerca, sostituzione e lettura di sottomoduli per percorso puntato (ad esempio
  `model.layers.3.mlp`), sostituzione di parametri;
- `indexing.py`: lettura dei dizionari dei modelli in modo generico (`flatten_index_paths`, `get_block_config`,
  `num_blocks`);
- `batches.py`: normalizzazione dei batch di calibrazione (`normalize_batch`, `pack_batch`);
- `devices.py`, `progress.py`: dispositivi e barre di avanzamento.

Il resto è stato spostato nella cartella a cui appartiene: gestore, schemi, gruppi, calibrazione a cascata e
configurazione in `compression/`; la conversione per l'export in `export/convert.py`.

### 4.2 `models/`

**Prima.** Ogni famiglia aveva un file `define_<modello>.py` con tre classi scritte a mano (configurazione,
modello, gestore), quasi identiche tra le famiglie (otto copie).

**Dopo.** Ogni famiglia è una cartella con **due dizionari**:

- il **dizionario del modello** (`indexing_<m>.py`): dove sono i layer, quali si possono comprimere, come la
  potatura li accoppia, come calibrarli, e quali operazioni servono al grafo di export (`structure`, `attn_type`,
  `mlp_type`, `norm_type`, `rope_type`);
- il **dizionario di famiglia** (`family_<m>.py`): quali classi Hugging Face la famiglia estende e i nomi delle
  classi generate. Le tre classi vengono generate da `models/factory.py::build_family`.

Esempio (famiglia Llama):

```python
LLAMA_C_FAMILY = {
    "indexing": LLAMA_C_INDEXING,
    "hf_config": "transformers.models.llama.configuration_llama.LlamaConfig",
    "hf_model": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
    "class_names": {"config": "LlamaConfigCompress", "model": "LlamaForCausalLMCompress",
                    "manager": "LlamaCompressionSchemesManager"},
    "auto_class": "AutoModelForCausalLM",
}
```

### 4.3 `compression/`

Contiene ora il **gestore** (`manager.py`, la classe con cui l'utente configura e applica la compressione), gli
**schemi** (`scheme.py`: uno per layer comprimibile, con le sue opzioni), i **gruppi** di schemi che devono
condividere la stessa maschera di potatura (`grouping.py`), la calibrazione a cascata (`cascade.py`, in cui i
blocchi vengono calibrati uno alla volta sul modello già compresso fino a quel punto), il registro delle opzioni
(`registry.py`), i compressori (`lrd.py`, `structured_pruning.py`, `block_pruning.py`, `unstructured_pruning.py`,
`quantization.py`, `smoothquant.py`) e le varianti di ogni metodo (`*_methods/`).

Cambiamenti principali:

- **La topologia è dello schema, non dei compressori.** La potatura strutturata deve sapere quali layer ricevono
  le uscite del layer potato, anche attraverso i layer di normalizzazione e verso il blocco successivo. Questa
  logica (con i nomi speciali `'preprocessing'`, `'final_layer'`, `'final_norm'`, detti *sentinelle*) era dentro
  il compressore. Ora è in `CompressionScheme.coupled_target_chains()` e `resolve_in_block()`: i compressori
  ricevono catene di moduli già risolte e non leggono più percorsi.
- **Un solo punto conosce lo stato interno della quantizzazione.** La quantizzazione lascia sui layer alcuni
  attributi privati (scale, precisione, parametri delle attivazioni). Prima venivano letti direttamente anche da
  `export/`. Ora solo `compression/quantization.py` li legge, ed espone due funzioni pubbliche:
  `export_quant_info(module)`, che descrive la quantizzazione di un layer in modo indipendente dal backend, e
  `copy_quant_state(old, new)`, che la copia sul layer convertito.
- **Validatori delle opzioni costruiti da fabbriche.** Circa venticinque funzioni di validazione scritte a mano
  ripetevano cinque schemi (valore in un insieme, numero in un intervallo, numero positivo, intero positivo,
  booleano). Ora sono costruite dalle funzioni di `compression/validators.py`.
- **L'elenco dei metodi validi deriva dal registro dei metodi.** Ad esempio i metodi di decomposizione validi sono
  le chiavi di `lrd_methods/__init__.py::METHOD_FUNCTIONS`: aggiungere un metodo richiede un file e una riga.

### 4.4 `calibration/`

Contiene la raccolta dei dati grezzi (`raw_data/`, funzioni agganciate al passaggio in avanti o all'indietro del
modello), i riepiloghi statistici (`summaries/`, ad esempio covarianza e intervallo delle attivazioni), il
calcolo del valore assoluto massimo per canale usato da SmoothQuant (`input_absmax.py`, prima dentro
`compression/`) e, nuovo, `datasets.py`: il caricatore delle finestre di testo per la calibrazione (WikiText-2),
che prima era copiato in otto script.

SmoothQuant è una tecnica che sposta parte della variabilità delle attivazioni nei pesi, moltiplicando le
attivazioni per un fattore e dividendo i pesi per lo stesso fattore, così che entrambi si quantizzino meglio.

### 4.5 `blocks/`

**Prima.** Pochi file grandi (`mha.py`, `decoder.py`, `norm.py`, ...) in cui le varianti erano scelte con
parametri booleani e catene di `if`/`elif`, e i blocchi leggevano la configurazione di compressione.

**Dopo.** Il transformer semplificato è diviso in **atomi**, un componente per file, raggruppati per famiglia:

```
blocks/
├── layers/      layer comprimibili: LinearCompressed, EmbeddingCompressed, Conv2d/3dCompressed, VCONBlock
├── norm/        rmsnorm, rmsnorm_prescaled, rmsnorm_upcast, layernorm
├── mlp/         plain (due lineari), gated (tre lineari)
├── positions/   rotazione delle posizioni (RoPE) e tabelle delle frequenze
├── attention/   base, causal (decoder), bidirectional (encoder)
│   ├── kernels/     manual, sdpa, custom_sdpa, paged
│   └── kv_cache/    mutable, io_scatter, io_concat, io_inplace, io_paged
└── stacks/      decoder, encoder (la pila di blocchi completa)
```

RoPE (*rotary position embedding*) è il modo in cui molti decoder codificano la posizione di ogni parola:
ruotano chiavi e interrogazioni dell'attenzione di un angolo che dipende dalla posizione.

Gli atomi sono elencati nel **catalogo dei blocchi**, `blocks/catalog.py::BLOCK_CATALOG` (sezione 5.1). I blocchi
non leggono più la configurazione di compressione: `convert_for_export` legge dimensioni, rango e bias dai layer
già compressi del modello Hugging Face.

### 4.6 `export/`

**Prima.** Un file di 1180 righe (`export/common.py`) con tutte le funzioni condivise, un registro di funzioni
`export_with_<backend>` e tre modi diversi di interpretare l'ingresso dell'export.

**Dopo.** Tre livelli:

1. **passi generici** comuni a ogni backend: conversione al grafo di `blocks/` (`convert.py`), involucro
   tracciabile (`wrapper.py`), lettura della quantizzazione (`quant_info.py`), quantizzazione con il flusso PT2E
   di PyTorch (`pt2e.py`; PT2E, *PyTorch 2 Export*, è il meccanismo con cui PyTorch inserisce operazioni di
   quantizzazione in un grafo tracciato), controlli (`checks.py`), *manifest* (`manifest.py`: un file JSON scritto
   accanto al file esportato, che descrive forme e nomi degli ingressi e delle uscite, così che chi lo esegue non
   debba ricostruirli) e risultato (`result.py`);
2. **il dizionario delle operazioni**, `operations.py::BACKEND_OPERATIONS` (sezione 5.2);
3. **un esportatore per backend** in `backends/`: ognuno è una sottoclasse di `Exporter` registrata con
   `@register_exporter`; `export_to_backend(model, config)` sceglie l'esportatore dal nome del backend.

L'esportatore riconosce gli atomi che richiedono un trattamento particolare (ad esempio il nucleo
dell'attenzione, che QNN mantiene in virgola mobile) da un attributo di classe `EXPORT_ROLE` dichiarato
dall'atomo, non più dal nome della classe.

### 4.7 `hf/`

`export_to_hf` salva il modello in una cartella caricabile con il codice della libreria (e, se richiesto, lo
pubblica su Hugging Face Hub). La voce `auto_map` del file di configurazione ora deriva dalla chiave
`auto_class` del dizionario di famiglia invece che da un elenco fisso.

### 4.8 Script, esempi e benchmark

- Il caricatore di WikiText-2 era copiato in sei script ed esempi e in due benchmark: ora usano tutti
  `calibration/datasets.py` (verificato: i campioni prodotti sono identici a quelli del vecchio caricatore).
- `scripts/executorch/qnn/full_quant_export.py` era per circa il 90 % una copia di
  `exporter_function_test.py`: le due differenze (quali layer quantizzare, metodi delle attivazioni) sono ora
  opzioni dello stesso script (`--quant-layers {none,mlp,all}`, `--quant-method`, `--act-method`,
  `--act-scheme`); le opzioni del vecchio script sono accettate come alias.

---

## 5. I due dizionari

### 5.1 `BLOCK_CATALOG`: che cosa serve al modello

Per ogni atomo il catalogo indica dove è implementato (`object`), che cosa fa (`summary`), quale operazione
fornisce (`implements`: atomi con lo stesso valore sono implementazioni intercambiabili della stessa operazione)
e chi lo sceglie (`selected_by`): il **modello**, perché ha bisogno di quell'operazione, o l'**export**, perché il
dispositivo preferisce quell'implementazione.

| Famiglia | Atomo | Operazione | Scelto da | Che cosa fa |
|---|---|---|---|---|
| stack | `transformer_decoder` | pila decoder | modello | blocchi pre-normalizzati, rotazione delle posizioni, cache KV, normalizzazione finale |
| stack | `transformer_encoder` | pila encoder | modello | blocchi pre- o post-normalizzati, senza cache |
| attention | `mha_causal` / `mha_encoder` | attenzione | modello | attenzione causale con cache / bidirezionale senza cache |
| attention_kernel | `manual`, `sdpa`, `custom_sdpa`, `paged` | calcolo dell'attenzione | export | prodotti espliciti; funzione di PyTorch; kernel fuso di ExecuTorch; kernel con cache a pagine |
| kv_cache | `mutable`, `io_scatter`, `io_concat`, `io_inplace`, `io_paged` | cache KV | export | stato interno; ingresso/uscita del grafo aggiornato in modi diversi; a pagine |
| mlp | `mlp`, `mlp_gated` | rete feed-forward | modello | due lineari; tre lineari con porta moltiplicativa |
| norm | `rmsnorm`, `rmsnorm_prescaled`, `rmsnorm_upcast` | normalizzazione RMS | export | semplice; riscalata prima per evitare traboccamenti in mezza precisione; calcolata in precisione singola |
| norm | `layernorm` | normalizzazione di layer | modello | — |
| rope_frequencies | `default`, `llama3` | frequenze RoPE | modello | semplici; scalate per bande (contesti lunghi) |

Alcuni termini della tabella: un blocco è *pre-normalizzato* se la normalizzazione è applicata prima
dell'attenzione e della rete feed-forward, *post-normalizzato* se dopo; un kernel *fuso* esegue in un'unica
operazione più passaggi che altrimenti sarebbero separati; una cache *a pagine* è divisa in porzioni di
dimensione fissa usate man mano che la sequenza cresce; la *porta moltiplicativa* è il terzo lineare della rete
feed-forward, la cui uscita (dopo l'attivazione) moltiplica elemento per elemento quella del secondo.

La normalizzazione RMS (*root mean square*) divide ogni vettore per la radice della media dei quadrati dei suoi
elementi. In mezza precisione (numeri a 16 bit) il quadrato di valori grandi può superare il massimo
rappresentabile: per questo esistono le varianti riscalata e in precisione singola.

### 5.2 `BACKEND_OPERATIONS`: come lo esegue il dispositivo

Per ogni tipo di blocco e ogni backend il dizionario indica l'implementazione **predefinita** e le **alternative
supportate**, ognuna con una frase che descrive l'operazione sul dispositivo. Indica anche come arriva la scelta
al codice: un'opzione della conversione (`convert_option`), un campo della configurazione del backend
(`config_field`), oppure il dizionario del modello (`model_key`, per i blocchi che l'export non sceglie).

| Tipo di blocco | Come si sceglie | XNNPACK | QNN | ONNX | TensorRT |
|---|---|---|---|---|---|
| cache KV | `cache_impl` | `mutable` | `mutable` | `io_inplace` | `io_inplace` |
| calcolo dell'attenzione | `attn_impl` | `manual` | `manual` | `manual` | `manual` |
| softmax mascherata | `masked_softmax` | no | no | no | no |
| normalizzazione RMS | `rmsnorm_impl` | `rmsnorm_prescaled` | `rmsnorm_prescaled` | `rmsnorm_prescaled` | `rmsnorm_prescaled` |
| dimensione di batch | `add_batch_dim` | no | no | no | no |
| quantizzazione dei lineari | campo di configurazione | `pt2e` | `linear` | `qdq` | `dequantize` |
| esecuzione dei lineari | campo di configurazione | fully-connected | FullyConnected | Gemm | prodotto di matrici |
| rete feed-forward | modello (`mlp_type`) | — | — | — | — |
| frequenze RoPE | modello (`rope_type`) | — | — | — | — |
| embedding, testa di uscita | implementazione unica | unica | unica | unica | unica |

Legenda dei valori: la *softmax* trasforma i punteggi dell'attenzione in pesi che sommano a uno; nella variante
*mascherata* le posizioni escluse vengono sostituite da un valore basso invece di ricevere una penalità additiva.
Per la quantizzazione dei lineari: `pt2e` usa il flusso PT2E (sezione 4.6); `linear` (QNN) quantizza solo i
lineari compressi, `all` (alternativa) tutto il grafo; `qdq` inserisce nel file ONNX coppie di operazioni
"quantizza/dequantizza"; `dequantize` (TensorRT) riconverte i pesi interi in virgola mobile prima del prodotto.
*Fully-connected*, *FullyConnected* e *Gemm* sono i nomi che XNNPACK, QNN e ONNX danno al layer lineare. Il
trattino indica una scelta che spetta al dizionario del modello; "unica" indica che esiste una sola
implementazione.

Esempio di voce (cache KV, solo due backend):

```python
"kv_cache": {
    "description": "Storage and update of past keys/values (blocks/attention/kv_cache).",
    "convert_option": "cache_impl",
    "backends": {
        "xnnpack": {"default": "mutable", "options": {
            "mutable": "ExecuTorch mutable buffer updated in place (index_copy_).",
            "io_scatter": "Cache as graph input/output, functional index_put.",
            ...}},
        "onnx": {"default": "io_inplace", "options": {
            "io_inplace": "ONNX TensorScatter (opset 24); consumers may update the buffer in place.",
            ...}},
    },
},
```

Gli esportatori leggono questo dizionario: le opzioni non date dall'utente prendono il valore predefinito del
backend (`resolve_convert_options`), e una scelta non elencata viene rifiutata prima di iniziare il tracciamento.
La scelta esplicita da parte dell'utente delle alternative non predefinite è prevista ma non ancora progettata
(sezione 11).

---

## 6. Duplicazioni eliminate

| Prima (copie) | Ora (una sola definizione) |
|---|---|
| inizializzazione e annullamento della decomposizione in `LinearCompressed` ed `EmbeddingCompressed` | `blocks/layers/low_rank.py::LowRankMixin` |
| costruzione dei lineari ripetuta nell'attenzione e nelle due reti feed-forward | `blocks/layers/linear_compressed.py::linear_from_spec` |
| tre varianti della normalizzazione RMS in una classe con parametri booleani | un atomo per file e una funzione comune `rms_normalize` |
| scrittura della cache e calcolo dell'attenzione con catene `if`/`elif` | atomi `kv_cache/*` e `kernels/*` con un'interfaccia comune |
| configurazione della pila convertita, separata per decoder ed encoder | `blocks/stacks/config.py` |
| otto file `define_*.py` quasi identici | `models/factory.py::build_family` e i dizionari di famiglia |
| `export/common.py` (1180 righe) | sei moduli con un compito ciascuno (sezione 4.6) |
| tre interpretazioni dell'ingresso dell'export; normalizzazione dei dispositivi fatta due volte | `export/exporter.py::resolve_components` e `Exporter.export` |
| scrittura del file `.pte` e passi finali ripetuti in XNNPACK (due volte) e in QNN | `ExecuTorchExporter.write_program` e `finalize` |
| elenco dei diciannove attributi di quantizzazione in due file | `QUANT_RUNTIME_BUFFERS` e `QUANT_RUNTIME_ATTRS` |
| estrazione della scala di quantizzazione copiata in due file | `compression/quantization.py::extract_torchao_scale` |
| simulazione della quantizzazione delle attivazioni in ingresso e in uscita; regola dei limiti interi ripetuta tre volte | `_make_fake_quant_hook` e `activation_qrange` |
| raccoglitori e riepiloghi separati per pesi e bias, due riepiloghi di intervallo | `ParamCollector`, `_ParamSummary`, `_RangeSummary` |
| costruzione a mano dei batch in cinque punti della calibrazione a cascata | `utils/batches.py` |
| lettura della configurazione di un blocco ripetuta in sei punti | `utils/indexing.py::get_block_config` e `num_blocks` |
| raccolta del massimo per canale di SmoothQuant dentro `compression/` | `calibration/input_absmax.py` |
| verifica del tipo di compressore dentro il gestore | metodo generico `Compressor.prepare_for_save` |
| nome fisso `"text"` del componente da esportare | il componente con `structure == "transformer_decoder"` |
| caricatore WikiText-2 copiato in otto script | `calibration/datasets.py` |
| `full_quant_export.py` (copia al 90 %) | `exporter_function_test.py --quant-layers all` |
| circa venticinque validatori scritti a mano | `compression/validators.py` |
| nomi degli ingressi e delle uscite della cache in due file | `export/manifest.py::graph_io_names` |
| elenchi dei metodi validi ripetuti accanto al registro dei metodi | derivati da `METHOD_FUNCTIONS` |
| lettura degli attributi privati della quantizzazione in `export/` | `export_quant_info`, `copy_quant_state` |
| risoluzione dei percorsi dentro due compressori | `CompressionScheme.coupled_target_chains`, `resolve_in_block` |

---

## 7. Correzioni di comportamento

Le modifiche seguenti cambiano il comportamento della libreria. Sono tutte correzioni di errori emersi durante la
rifattorizzazione.

- **Layer completamente potato.** Un `LinearCompressed` con zero uscite non inizializzava correttamente la classe
  base di PyTorch e andava in errore se chiamato. Ora è un modulo valido che restituisce il
  proprio ingresso invariato, come previsto dalla sua documentazione.
- **Encoder convertiti.** La normalizzazione usava sempre la costante di stabilità `1e-5`, mentre BERT usa
  `1e-12`. Ora si usa quella del modello. Verificato: BERT, DistilBERT e ViT convertiti coincidono con Hugging
  Face con uno scarto massimo di `2e-5` (test `test/export/test_convert.py`).
- **Qwen2.5-VL.** La classe del modello non dichiarava la propria classe di configurazione (era scritta come
  annotazione di tipo invece che come assegnazione); la fabbrica ora la imposta.
- **Errori di importazione dei modelli.** `models/__init__.py` nascondeva qualunque eccezione durante
  l'importazione di una famiglia. Ora intercetta solo l'assenza della famiglia nella versione installata di
  `transformers`, con un avviso.
- **Cache a pagine.** Il dizionario di export garantisce che la cache a pagine (`io_paged`) sia sempre usata con
  il kernel a pagine (`paged`), l'unico che sa leggerla.
- **ONNX e TensorRT.** La cache predefinita era `mutable`, che questi backend non supportano (dava errore); ora è
  `io_inplace`.
- **Token di calibrazione.** L'opzione `calibration_token_ids` è disponibile anche per XNNPACK.

---

## 8. Codice archiviato

Il codice non più usato non è stato cancellato ma spostato in `archive/`, con una nota che spiega il motivo:

- `archive/side_projects/`: due progetti collaterali (`qwen-vl-finetune/`, `vlm-modules-importance/`) non usati
  dalla libreria;
- `archive/code/`: funzioni mai chiamate o non funzionanti, ognuna con una docstring che spiega perché è stata
  tolta (funzioni per la rotazione delle posizioni multimodale, una variante di attenzione con proiezioni fuse,
  rami dell'encoder per posizioni e suddivisione in tasselli mai usati, un vecchio percorso di quantizzazione,
  parti inutilizzate del vecchio `export/common.py`);
- `archive/scripts/`: lo script QNN sostituito (sezione 4.8);
- `archive/tests/`: la vecchia suite di test, aggiornata ai nuovi percorsi ed eseguibile
  (`pytest archive/tests`), conservata come riferimento di regressione.

Sono stati invece rimossi senza archiviarli elementi che non erano codice: un `__init__.py` alla radice che
conteneva solo un commento, cartelle di test che contenevano solo file compilati, metodi vuoti del gestore e
chiavi dei dizionari dei modelli che nessuna parte del codice leggeva.

---

## 9. Test

### 9.1 Organizzazione

La nuova suite in `test/` ricalca la struttura del pacchetto: una cartella per ogni cartella della libreria
(`utils`, `blocks`, `compression`, `calibration`, `models`, `export`, `hf`). Usa modelli minuscoli con pesi
casuali (`test/_helpers/model_factory.py`), quindi non scarica nulla. I test che richiedono un backend vengono
saltati automaticamente se il backend non è installato (`test/_helpers/capabilities.py`).

Oltre ai controlli sui singoli oggetti, la suite verifica **l'equivalenza numerica** tra il modello Hugging Face
e il grafo convertito (decoder con tre tipi di cache, encoder BERT, DistilBERT e ViT), la coerenza tra i due
dizionari e il catalogo, e un export reale per ogni backend disponibile.

### 9.2 Risultati

I test sono stati eseguiti in tre ambienti, ognuno con un insieme diverso di backend installati:

| Ambiente | Nuova suite | Vecchia suite (`archive/tests`) |
|---|---|---|
| `py312_tensorrt` (ONNX, TensorRT, scheda grafica) | 212 superati, 4 saltati | 145 superati, 18 saltati |
| `py312_executorch_xnnpack` | 210 superati, 6 saltati | 2 fallimenti d'ambiente, 1 errore di raccolta |
| `py310-executorch-qnn` (con l'SDK Qualcomm) | 210 superati, 6 saltati | 6 fallimenti d'ambiente, 1 errore di raccolta |

I fallimenti della vecchia suite sono **dovuti all'ambiente** ed erano presenti anche prima della
rifattorizzazione: mancano il pacchetto `onnx` (errore di raccolta), alcuni operatori quantizzati e il kernel
fuso di ExecuTorch, e un test si aspetta l'assenza dell'SDK Qualcomm che invece è installato. L'elenco esatto è in
`HANDOFF.md`, sezione 6. Nessun test è peggiorato.

Una nota sul test di export QNN della nuova suite: usa la quantizzazione di tutto il grafo
(`quant_scope="all"`), perché con l'impostazione predefinita (solo i lineari compressi) la versione installata del
compilatore Qualcomm rifiuta alcune operazioni in virgola mobile.

---

## 10. Verifica delle regole

### 10.1 "Un'aggiunta tocca una sola cartella"

| Aggiunta | File da toccare | Verifica |
|---|---|---|
| nuova famiglia di modelli | `models/<m>_c/` (dizionario del modello, dizionario di famiglia, `__init__.py`) e una riga in `models/__init__.py` | prova reale con Mistral: compressione, conversione e confronto con Hugging Face (scarto `9e-6`) senza toccare altro |
| nuovo metodo di decomposizione, quantizzazione o potatura non strutturata | un file in `compression/<tipo>_methods/` e una riga nel suo `METHOD_FUNCTIONS` (più `CALIBRATED_METHOD_DICT` se richiede calibrazione) | verificato sul codice |
| nuovo riepilogo di calibrazione | un file in `calibration/summaries/` e una riga in `SUMMARY_REGISTRY` (più un raccoglitore in `raw_data/` se servono dati nuovi) | verificato sul codice |
| nuovo backend | un modulo in `export/backends/` (sottoclasse di `Exporter`), un'importazione, e la colonna del backend in `BACKEND_OPERATIONS` | verificato sul codice |

### 10.2 Eccezioni motivate

- **Atomi legati a un backend in `blocks/`.** Il kernel `custom_sdpa` usa un operatore di ExecuTorch e il kernel
  `paged` segue il contratto del plugin di attenzione di TensorRT Edge-LLM (la libreria di NVIDIA per eseguire
  modelli linguistici su dispositivi embedded come le schede Jetson). Sono ammessi perché sono atomi selezionabili: il
  resto di `blocks/` non dipende da essi, e il dizionario di export li propone solo ai backend che li supportano.
- **La potatura a blocchi modifica un attributo del modello Hugging Face** (il numero di teste per gruppo) per
  mantenerlo eseguibile dopo la potatura. Non viola la regola: il compressore è generico (blocchi di righe
  raggruppati in unità) e i nomi degli attributi da leggere e aggiornare vengono dal dizionario del modello.
- **`export_to_hf` applica la compressione** se riceve il gestore. È l'unico punto in cui un passo di export
  avvia la compressione; è stato mantenuto per compatibilità con chi lo usa ed è documentato.
- **Nomi di modelli fuori da `models/`.** Non esiste più alcun ramo di codice specifico per un modello fuori da
  `models/`, e i commenti descrivono la caratteristica (ad esempio "un modulo di embedding composito") invece del
  modello. Resta solo `llama::custom_sdpa`, che è il nome di un operatore di ExecuTorch.

---

## 11. Limiti e lavoro futuro

- **Scelta delle alternative da parte dell'utente.** Il dizionario di export elenca le implementazioni supportate
  e quella predefinita; oggi l'utente può già scegliere passando le opzioni di conversione, ma un'interfaccia
  dedicata (ad esempio profili per dispositivo) è ancora da progettare.
- **Modelli con proiezioni fuse.** ModernBERT calcola interrogazioni, chiavi e valori con un'unica matrice; il
  grafo di export non ha ancora un atomo per questa forma, quindi ModernBERT si comprime ma non si esporta.
- **Modelli visione-linguaggio.** Le famiglie Qwen2-VL e Qwen2.5-VL si comprimono, ma il loro dizionario non
  dichiara una `structure` per la parte visiva, che quindi non viene convertita per l'export.
- **Vecchia suite di test.** È conservata come riferimento; quando la nuova suite sarà considerata sufficiente si
  può decidere di eliminarla.
- **Embedding quantizzati.** La tabella degli embedding resta in virgola mobile anche negli export quantizzati;
  nei modelli con vocabolari grandi è la parte più pesante del file.
