# Adaptive Temperature for Mathematical Reasoning in LLMs: Implementation of Temperature Scaling Papers

This repository contains Colab notebook implementations exploring adaptive temperature scaling mechanisms for improving mathematical reasoning in Large Language Models (LLMs). The implementation is based on insights from two key papers:

1. ["Softmax Is Not Enough: Theory-Guided Temperature Adaptation for Language Models"](https://arxiv.org/abs/2311.11298)
2. ["Temperature Scaling as Adaptive Temperature Control: An Information-Geometric View"](https://arxiv.org/abs/2410.01104)

The implementation focuses on dynamically adjusting the temperature parameter based on prediction entropy to enhance model performance on mathematical reasoning tasks.

## Paper Implementation Details

The implementation synthesizes key ideas from both papers:
- Dynamic temperature adaptation based on prediction entropy
- Polynomial control function for temperature scaling
- Information-geometric perspective on temperature adaptation
- Token-by-token analysis and adjustment

### Theory and Implementation
The core concepts include:
- Standard softmax-based sampling with fixed temperature is suboptimal for complex reasoning tasks
- Temperature should be dynamically adjusted based on token prediction entropy
- A polynomial control function maps entropy to temperature scaling (beta)
- Scaling is triggered when entropy exceeds a threshold
- Information geometry provides insights into optimal temperature control

## Notebooks

### 1. Small-Scale Evaluation (n=50)
- Uses 50 samples from GSM8K dataset
- Useful for quick experimentation and parameter tuning
- Shows better performance with adaptive temperature in some runs
- Good for understanding the papers' concepts

### 2. Large-Scale Evaluation (n=200)
- Uses 200 samples from GSM8K dataset
- Provides more robust evaluation metrics
- Current implementation shows baseline outperforming adaptive temperature
- Useful for benchmarking against papers' results

## Getting Started

### Quick Setup
Run these commands in either notebook:
```bash
!pip install datasets transformers torch tqdm
!huggingface-cli login
```

### Requirements
- Google Colab (GPU runtime recommended)
- Hugging Face account and access token
- Access to required models

## Hyperparameters

Both notebooks use this configuration, with parameters based on the papers:
```python
class Config(NamedTuple):
    model_name: str = "google/gemma-2-2b-it"  # Model to use
    entropy_threshold: float = 0.3    # Triggers adaptive temperature
    poly_coeffs: Tuple[float, ...] = (
        -1.791, 4.917, -2.3, 0.481, -0.037
    )                                 # Temperature control coefficients
    max_new_tokens: int = 500        # Max generation length
    max_tokens: int = 2048           # Total token limit
    top_p: float = 0.9              # Nucleus sampling
    top_k: int = 40                 # Top-k sampling
    num_samples: int = 50 or 200    # Varies by notebook
    min_beta: float = 0.5           # Min inverse temperature
```

### Key Parameters from Papers
- **entropy_threshold**: Controls when adaptive scaling activates (H > θ)
- **poly_coeffs**: Temperature adaptation curve coefficients: β(H) = sum(ai * H^i) for i=0 to 4
- **min_beta**: Sets minimum scaling factor (β_min) for stability

## Running the Notebooks

1. Choose the appropriate notebook based on your needs:
   - Use n=50 for quick experiments and parameter tuning
   - Use n=200 for more thorough evaluation
2. Execute setup cells
3. Log in to Hugging Face when prompted
4. Run main experiment
5. Check results in output

## Output

Both notebooks generate:
- Solutions using baseline and adaptive methods
- Performance comparisons
- Token-level statistics and entropy analysis
- Saved results in JSON format

## Known Results

### n=50 Notebook:
- Shows promising results for adaptive temperature
- Useful for parameter exploration
- Results align with papers' findings on shorter samples

### n=200 Notebook:
- Currently shows baseline outperforming adaptive temperature
- Provides more statistically significant results
- Differences from papers' results suggest need for further investigation

## Troubleshooting

If you encounter issues:
- Verify Hugging Face authentication
- Enable GPU runtime
- Confirm package installation
- Check model access permissions

## Notes
- Token metrics appear at execution end
- Results vary by model and dataset
- Uses GSM8K dataset for evaluation
- Implementation synthesizes concepts from both papers
- Performance differences between sample sizes suggest need for further investigation

## Paper References
This implementation is based on:
```
@article{softmax-not-enough,
  title={Softmax Is Not Enough: Theory-Guided Temperature Adaptation for Language Models},
  journal={arXiv preprint arXiv:2311.11298},
  year={2023}
}

```

## Contributing
Feel free to contribute to improving this implementation or exploring different parameter settings. The current results suggest there's room for optimization and better alignment with the papers' findings.
