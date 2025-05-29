# MathBatch: Multimodal Mathematical Reasoning Batch Generation Framework

MathBatch is a modular framework for batch processing of mathematical reasoning tasks using multimodal large models. Designed for researchers and developers, it efficiently handles both text-only and multimodal (text+image) datasets with a flexible, extensible architecture.

## Features

- **Multimodal Processing**: Handles both text-only and text+image mathematical reasoning tasks
- **Batch Processing**: Optimized for large-scale datasets with efficient batch inference
- **Modular Design**: Easily extend or replace components with custom implementations
- **Checkpoint System**: Automatic saving and resuming of processing jobs

## Usage

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

### Key Parameters:

**Model Configuration:**
- `--model_path`: Path to model weights directory (required)
- `--model_name`: Model identifier name (required)
- `--tensor_parallel_size`: GPU parallelization (default: 4)
- `--max_model_len`: Maximum context length (default: 32768)

**Generation Parameters:**
- `--temperature`: Sampling temperature (default: 0.0)
- `--top_p`: Top-p sampling value (default: 1.0)
- `--max_tokens`: Maximum tokens to generate (default: 1024)

**Dataset Parameters:**
- `--multimodal_input`: Input JSON file for multimodal data
- `--multimodal_output`: Output file for multimodal results
- `--text_only_input`: Input JSON file for text-only data
- `--text_only_output`: Output file for text-only results
- `--image_base_path`: Base directory for images (required for multimodal)

**Field Mapping:**
- `--index_field`: Unique identifier field (default: "sample_index")
- `--query_field`: Field containing prompts (default: "query_cot") 
- `--image_field`: Field containing image paths (default: "image")
- `--response_field`: Output field for responses (default: "model_answer")

**Processing Control:**
- `--batch_size`: Number of samples per batch (default: 500)
- `--rerun`: Reprocess all samples (ignore existing results)

## Customization

MathBatch's modular design allows easy extension through abstract base classes:

### Custom Data Processing Example

```python
class MathProblemPreprocessor(BaseDataProcessor):
    """Custom processor for mathematical problem datasets"""
    
    def __init__(self, query_field: str, image_base_path: str):
        super().__init__(query_field)
        self.image_base_path = image_base_path
        
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Enhances math problems with step-by-step reasoning prompts"""
        prompt = f"Solve this problem step by step: {item[self.query_field]}"
        image_path = os.path.join(self.image_base_path, item.get("diagram", ""))
        
        return {
            "prompt": prompt,
            "image": Image.open(image_path) if os.path.exists(image_path) else None
        }
```

### Custom Model Integration Example

```python
class CustomMathModel(BaseInferenceModel):
    """Wrapper for custom mathematical reasoning model"""
    
    def __init__(self, model_path: str):
        self.model = load_custom_model(model_path)
        
    def generate_responses(self, prompts: List[str], images: List[Optional[Image.Image]]) -> List[str]:
        """Generates step-by-step solutions for math problems"""
        return [self.model.solve(prompt) for prompt in prompts]
```

### Extensible Components

MathBatch provides these abstract base classes for customization:

1. **Data Processing**:
   - `BaseDataLoader`: Custom data loading logic
   - `BaseDataProcessor`: Data preprocessing pipelines
   - `BaseDataFilter`: Filtering unprocessed items
   - `BaseBatchProcessor`: Custom batching strategies
   - `BaseDataSaver`: Output formatting and saving

2. **Model Integration**:
   - `BaseInferenceModel`: Interface for custom MLMs

## Project Structure

```
├── data_processing/
│   ├── base_processor.py    # Abstract base classes
│   ├── batch_processor.py   # Batch processing logic
│   ├── data_filter.py       # Data filtering
│   ├── data_loader.py       # Dataset loading
│   ├── data_processor.py    # Text/multimodal processors
│   ├── data_saver.py        # Result saving
│   └── pipeline_executor.py # Main pipeline
│
├── models/
│   ├── base_model.py        # Model interface
│   └── vllm_model.py        # vLLM implementation
│
└── generate_response.py     # Main executable
```
