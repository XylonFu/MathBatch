# MathBatch: Multimodal Mathematical Reasoning Batch Generation Framework

MathBatch is a modular framework for batch processing of mathematical reasoning tasks using multimodal large models.
Designed for researchers and developers, it efficiently handles both text-only and multimodal (text+image) datasets with
a flexible, extensible architecture.

## Features

- **Multimodal Processing**: Handles both text-only and text+image mathematical reasoning tasks
- **Batch Processing**: Optimized for large-scale datasets with efficient batch inference
- **Modular Design**: Easily extend or replace components with custom implementations
- **Checkpoint System**: Automatic saving and resuming of processing jobs

## Usages

### Multimodal Processing

```bash
python generate_response.py \
  --model_path /path/to/model \
  --model_name your_model_identifier \
  --multimodal_input multimodal_data.json \
  --multimodal_output multimodal_results.json \
  --image_base_path /path/to/images \
  --index_field sample_index \
  --query_field query_cot \
  --image_field image \
  --response_field model_answer
```

### Text-Only Processing

```bash
python generate_response.py \
  --model_path /path/to/model \
  --model_name your_model_identifier \
  --text_only_input text_data.json \
  --text_only_output text_results.json \
  --index_field sample_index \
  --query_field query_cot \
  --response_field model_answer
```

### Key Parameters

**Model Configuration**:

- `--model_path`: Path to model weights directory (required)
- `--model_name`: Model identifier name (required)
- `--tensor_parallel_size`: GPU parallelization (default: 4)
- `--max_model_len`: Maximum context length (default: 32768)

**Generation Parameters**:

- `--temperature`: Sampling temperature (default: 0.0)
- `--top_p`: Top-p sampling value (default: 1.0)
- `--max_tokens`: Maximum tokens to generate (default: 1024)

**Dataset Parameters**:

- `--multimodal_input`: Input JSON file for multimodal data
- `--multimodal_output`: Output file for multimodal results
- `--text_only_input`: Input JSON file for text-only data
- `--text_only_output`: Output file for text-only results
- `--image_base_path`: Base directory for images (required for multimodal)

**Field Mapping**:

- `--index_field`: Unique identifier field (default: "sample_index")
- `--query_field`: Field containing prompts (default: "query_cot")
- `--image_field`: Field containing image paths (default: "image")
- `--response_field`: Output field for responses (default: "model_answer")

**Processing Control**:

- `--batch_size`: Number of samples per batch (default: 500)
- `--rerun`: Reprocess all samples (ignore existing results)

### Extensible Components

**Data Processing**:

- `BaseDataLoader`: Custom data loading logic
- `BaseDataFilter`: Filtering unprocessed items
- `BaseDataProcessor`: Data preprocessing pipelines
- `BaseDataBatcher`: Custom batching strategies
- `BaseDataSaver`: Output formatting and saving

**Model Integration**:

- `BaseInferenceModel`: Interface for custom MLMs

## Project Structure

```
├── cores/
│   ├── base.py               # Abstract base classes
│   ├── batcher.py            # Batch processing logic
│   ├── executor.py           # Main pipeline
│   ├── filter.py             # Dataset loading
│   ├── loader.py             # Data filtering
│   ├── processor.py          # Text/multimodal processors    
│   └── saver.py              # Result saving
│
├── models/
│   ├── base.py               # Model interface
│   └── vllm.py               # vLLM implementation
│
└── generate_response.py      # Main executable
```
