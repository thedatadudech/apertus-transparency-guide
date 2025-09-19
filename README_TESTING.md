# 🇨🇭 Swiss German AI Testing Scripts

Zwei Test-Scripts um die verschiedenen Modelle auf ihre Schweizerdeutsch-Fähigkeiten zu testen.

## 📋 Scripts Übersicht

### 1. `quick_tokenizer_test.py` - Schnelle Tokenizer-Analyse
**⚡ Schnell und lightweight**
- Nur Tokenizer-Loading (keine Models)
- Vergleicht 5+ verschiedene Tokenizer
- Zeigt Effizienz und Probleme
- Läuft auch auf CPU in ~30 Sekunden

```bash
python quick_tokenizer_test.py
```

### 2. `test_swiss_german_generation.py` - Vollständige Text-Generation
**🧠 Komplett aber ressourcenintensiv**
- Lädt komplette Models 
- Echte Text-Generation
- Speichert Ergebnisse als JSON
- Braucht GPU für große Models

```bash
python test_swiss_german_generation.py
```

## 🎯 Was getestet wird

### Test-Texte:
- **Swiss German 1**: `"Grüezi! Chönd Sie mer bitte d Schwyzer KI erchläre?"`
- **Swiss German 2**: `"Was isch KI und wie funktioniert das?"`
- **Standard German**: `"Hallo! Können Sie mir bitte die Schweizer KI erklären?"`
- **Swiss Dialect**: `"Mir händ hüt es schöns Wätter, gäll?"`
- **Technical German**: `"Die Künstliche Intelligenz verwendet neuronale Netzwerke."`

### Modelle:
- 🇨🇭 **Apertus Swiss AI** (`swiss-ai/Apertus-8B-Instruct-2509`)
- 🇩🇪 **German BERT** (`bert-base-german-cased`)
- 🇩🇪 **German GPT-2** (`dbmdz/german-gpt2`)
- 🌍 **Multilingual BERT** (`bert-base-multilingual-cased`)
- 🤖 **Standard GPT-2** (`gpt2`)

## 📊 Was analysiert wird

### Tokenizer-Qualität:
- **Tokens pro Zeichen** (niedriger = effizienter)
- **UTF-8 Encoding Probleme** (`Ã¼`, `Ã¶`, `Ã¤`)
- **Einzelzeichen-Tokens** (ineffizient)
- **Morphologie-Splits** (Compound-Behandlung)

### Text-Generation Qualität:
- **Schweizerdeutsch Authentizität**
- **Grammatikalische Korrektheit**
- **Kulturelle Angemessenheit**
- **Generierungs-Geschwindigkeit**

## 🚀 Empfohlener Ablauf

### Schritt 1: Quick Test
```bash
# Schneller Überblick über alle Tokenizer
python quick_tokenizer_test.py
```

### Schritt 2: Detaillierte Tests (wenn GPU verfügbar)
```bash
# Vollständige Generation-Tests
python test_swiss_german_generation.py
```

### Schritt 3: Remote Server Test
```bash
# Auf dem Remote Server mit GPU
ssh apertus
cd /workspace/apertus-transparency-guide
source .venv/bin/activate
python test_swiss_german_generation.py
```

## 📁 Output Files

### `quick_tokenizer_test.py`:
- Console Output mit Rankings
- Detaillierte Token-Aufschlüsselung

### `test_swiss_german_generation.py`:
- JSON File: `swiss_german_test_results_YYYYMMDD_HHMMSS.json`
- Enthält alle Generationen, Timings, Fehler

## 🔍 Interpretation der Ergebnisse

### Tokenizer Rankings:
- **Niedriger tok/char Ratio** = effizienter
- **Wenig "Ã" tokens** = bessere UTF-8 Behandlung
- **Wenig Einzelzeichen** = bessere Compound-Behandlung

### Generation Quality:
- **Authentisches Schweizerdeutsch** vs. Standard Deutsch
- **Konsistente Grammatik**
- **Kulturell angemessene Begriffe**

## ⚠️ Hardware Requirements

### Quick Test:
- ✅ CPU only
- ✅ 4GB RAM minimum
- ✅ ~2GB Download (Tokenizer)

### Full Test:
- 🎮 GPU empfohlen (8GB+ VRAM)
- 💾 16GB+ RAM  
- 📦 ~30GB Download (alle Models)

## 🐛 Troubleshooting

### "Model zu groß" Fehler:
```python
# In test_swiss_german_generation.py, reduziere max_new_tokens:
max_new_tokens=50  # statt 150
```

### UTF-8 Probleme:
```bash
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
```

### Memory Errors:
```python
# Verwende kleinere batch size oder float32 statt float16
torch_dtype=torch.float32
```

## 📈 Beispiel Output

```
🥇 German BERT        : 0.324 tok/char, 35 tokens, 2 problems
🥈 Apertus Swiss AI   : 0.315 tok/char, 34 tokens, 6 problems  
🥉 German GPT-2       : 0.306 tok/char, 33 tokens, 9 problems
4. Multilingual BERT  : 0.361 tok/char, 39 tokens, 3 problems
```

Das zeigt: **German BERT** ist am effizientesten mit wenigsten Problemen, aber **Apertus** ist überraschend gut bei Token-Effizienz!