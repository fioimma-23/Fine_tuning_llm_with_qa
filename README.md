# Domain-Specific Q\&A Generation via Instruction Fine-Tuning of Language Models   with Chatbot

## Objective

This project provides an **end-to-end pipeline** for fine-tuning a language model (Flan-T5 or LLaMA3) using structured text extracted from PDFs. The goal is to enhance performance on **domain-specific Q\&A tasks** by generating high-quality question-answer pairs and using efficient training methods.

---

## Pipeline Overview

### 🔁 Workflow

PDF → Markdown → Section Split → Q\&A Generation → Fine-Tuning → Inference

---

## Features

* 📄 **PDF to Markdown Conversion**: Maintains document structure using Marker PDF.
* 🔖 **Header-Based Splitting**: Organizes content into logical sections using Markdown headers.
* ❓ **Q\&A Generation**: Uses LLaMA3 via Ollama to generate question-answer pairs in Alpaca format.
* ⚙️ **Fine-Tuning**:

  * **Full Fine-Tuning**: Based on `Seq2SeqTrainer` for Flan-T5.
  * **PEFT (LoRA)**: Memory-efficient fine-tuning using 4-bit quantization and LoRA adapters.
* 🌐 **Inference**: Deployed using Streamlit for interactive querying of the fine-tuned model.

## Tools & Libraries

* Marker (for structured PDF parsing): [https://github.com/pytorch/marker](https://github.com/pytorch/marker)
* Ollama with LLaMA3: [https://ollama.com/](https://ollama.com/)
* Hugging Face Transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
* PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
* bitsandbytes: [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
* Others: Streamlit, pandas, numpy, tqdm

---

## Usage

### 1. PDF to Markdown

```bash
python scripts/pdf_to_md.py --input_dir pdf/ --output_dir markdown/
```

### 2. Split Markdown into Sections

```bash
python scripts/split_markdown_by_header.py --input_dir markdown/ --output_dir sections/
```

### 3. Generate Q\&A Pairs (Alpaca Format)

```bash
python scripts/generate_qa_pairs.py --input_dir sections/ --output_file qa_pairs/alpaca_data.json
```

### 4. Fine-Tune the Model

**Full Fine-Tuning (Flan-T5):**

```bash
python scripts/full_finetune_flan_t5.py --data qa_pairs/alpaca_data.json
```

**PEFT with LoRA (LLaMA3):**

```bash
python scripts/peft_lora_finetune.py --data qa_pairs/alpaca_data.json
```

### 5. Launch Inference App

```bash
streamlit run app/qa_inference_app.py
```

---

## Key Achievements

* ✅ **End-to-End Automation**: Fully automated Q\&A generation pipeline
* ⚡ **Efficiency**: Reduced VRAM usage via 4-bit quantization + LoRA
* 🧠 **High-Quality Data**: Structured Q\&A pairs generated using LLaMA3 + instruction format
