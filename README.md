# XAI-NIDS  An Interactive Quality-Driven Framework for Explainable 5G Network Intrusion Detection

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](README.md)

A comprehensive explainable AI (XAI) framework for network intrusion detection systems, featuring end-to-end ML workflows, multi-dimensional quality assessment, and LLM-powered explanations. Generate an interactive Dashboard for different stakeholders.


[![Dashboard Demo](thumbnail.png)](https://youtu.be/J88pDunxvtg)

Click the Dashboard Thumbnail to see the Dashboard Demo on YouTube!

## 🌟 Key Features

- **🔄 Complete ML Workflow**: End-to-end pipeline from data preprocessing to model evaluation
- **🎯 Multiple Model Support**: DecisionTree, RandomForest, XGBoost, DNN, MLP, LightGBM
- **🔍 Dual XAI Methods**: SHAP and LIME explanations with quality assessment
- **📊 Multi-Dimensional Evaluation**: Faithfulness, robustness, and complexity metrics
- **🤖 LLM-Powered Explanations**: Natural language explanations via Deepseek API
- **📈 Interactive Dashboard**: Web-based visualization of results and explanations
- **🌐 Multi-Dataset Support**: 5G-NIDD, 5GAD, 5GC_PFCP datasets
- **⚡ Production-Ready**: Optimized for both binary and multi-class classification

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    XAI Framework Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Phase 1: Data Processing & Feature Selection                │
│  └─ LightGBM + SHAP → Top-30 Features                       │
│                                                               │
│  Phase 2: Binary Classification                              │
│  ├─ Models: DT, RF, XGB, DNN, MLP                           │
│  ├─ XAI: SHAP + LIME explanations                           │
│  ├─ Quality: 3-dimensional evaluation                       │
│  └─ LLM: Global + individual explanations                   │
│                                                               │
│  Phase 3: Multi-class Classification                         │
│  ├─ Models: Same as Phase 2                                 │
│  ├─ XAI: Class-specific analysis                            │
│  └─ LLM: Attack-type explanations                           │
│                                                               │
│  Phase 4: Visualization & Synthesis                          │
│  └─ Interactive dashboard with all results                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for DNN training)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/XAI-Framework.git
   cd XAI-Framework
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # For LLM explanations
   export DEEPSEEK_API_KEY='your_api_key_here'
   
   # For evaluation sampling
   export XAI_EVAL_SAMPLE_SIZE=200
   export XAI_EVAL_STRATIFIED=1
   ```

### Running the Framework

#### Option 1: Complete Workflow
```bash
python workflow.py
```
This runs the entire pipeline: data processing, model training, XAI analysis, quality evaluation, and LLM explanations.

#### Option 2: LLM Explanations Only
```bash
python run_llm_explanations.py
```
Generate LLM explanations for existing model results.

#### Option 3: Interactive Dashboard
```bash
python server.py
# Open http://localhost:8080/index.html in your browser
```

## 📁 Project Structure

```
XAI-Framework/
├── workflow.py                    # Main pipeline orchestrator
├── evaluation_toolkit.py          # XAI quality assessment toolkit
├── llm_explanation_module.py      # LLM explanation generation
├── evaluation_config.yaml         # Configuration parameters
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LLM_Usage_Guide.md            # LLM feature guide
│
├── results/                       # Output directory
│   ├── 5G-NIDD/
│   ├── 5GAD/
│   └── 5GC_PFCP/
│       ├── binary/                # Binary classification results
│       │   ├── DecisionTree/
│       │   │   ├── confusion_matrix.png
│       │   │   ├── classification_report.png
│       │   │   ├── shap_summary_plot.png
│       │   │   ├── lime_explanation_*.html
│       │   │   ├── evaluate/      # XAI quality reports
│       │   │   └── LLM/           # LLM explanations
│       │   ├── RandomForest/
│       │   └── ...
│       └── multiclass/            # Multi-class results
│           └── ...
│
├── index.html                     # Dashboard frontend
├── app.js                        # Dashboard JavaScript
├── style.css                     # Dashboard styling
└── server.py                     # Local web server
```

## 🎯 XAI Quality Assessment

The framework evaluates XAI explanations across three core dimensions:

### 1. Faithfulness (40% weight)
Measures how accurately explanations reflect the model's decision-making process.

**Metrics:**
- **Deletion AUC**: Performance degradation when removing important features
- **Insertion AUC**: Performance improvement when adding important features
- **Feature Ablation Correlation**: Correlation between feature importance and impact

### 2. Robustness (35% weight)
Measures the stability of explanations under small perturbations.

**Metrics:**
- **Top-k Stability**: Jaccard similarity of top features under noise
- **Importance Variance**: Consistency of feature importance values


### 3. Complexity (25% weight)
Measures the simplicity and interpretability of explanations.

**Metrics:**
- **Sparsity**: Ratio of non-zero feature importances
- **Effective Complexity**: Distribution concentration of importance values


### Overall Score Calculation
```python
overall_score = (
    faithfulness_score * 0.40 +
    robustness_score * 0.35 +
    complexity_score * 0.25
)
```

**Quality Levels:**
- **0.9-1.0**: 🟢 Excellent - Production ready
- **0.7-0.9**: 🟡 Good - Suitable for deployment
- **0.5-0.7**: 🟠 Acceptable - Needs optimization
- **0.0-0.5**: 🔴 Poor - Major improvements required

## 🤖 LLM-Powered Explanations

The framework generates human-readable explanations using the Deepseek API.

### Setup

1. **Get API Key**: Visit [Deepseek Platform](https://platform.deepseek.com/)
2. **Set Environment Variable**:
   ```bash
   export DEEPSEEK_API_KEY='your_api_key_here'
   ```

### Features

- **Global Explanations**: Model behavior, feature importance, security implications
- **Individual Explanations**: Sample-specific analysis with threat assessment
- **Cybersecurity Focus**: Tailored for network security professionals
- **Efficient API Usage**: Caching, retry logic, optimized prompts

### Example Output

**Global Explanation:**
```markdown
# RandomForest Global Explanation

## Model Performance Analysis
The RandomForest model achieves 99.7% accuracy with excellent precision 
(0.998) and recall (0.996) for intrusion detection...

## Feature Importance Deep Dive
Top features: src_bytes (0.234), dst_bytes (0.189), duration (0.156)...

## Security Threat Coverage
Highly effective for detecting: SYN floods, port scans, DoS attacks...
```

**Individual Explanation:**
```markdown
## Sample 148299 - Malicious Traffic Analysis

### Traffic Pattern Analysis
This sample exhibits characteristics of a SYN flood attack with 
duration=0.05s, src_bytes=0, dst_bytes=1460...

### Threat Classification
Attack Type: SYN Flood | Confidence: 98.7%
Risk Severity: HIGH - Immediate response required
```

See [LLM_Usage_Guide.md](LLM_Usage_Guide.md) for detailed usage instructions.

## 📊 Interactive Dashboard

The framework includes a web-based dashboard for exploring results.

### Features

- **Multi-Phase Navigation**: Switch between workflow phases
- **Dataset Selection**: Compare results across datasets
- **Model Comparison**: Side-by-side model analysis
- **Interactive Visualizations**: Zoomable charts and plots
- **Integrated Explanations**: LLM-generated insights alongside XAI visuals

### Usage

```bash
python server.py
# Navigate to http://localhost:8080/index.html
```

### Dashboard Sections

1. **Overview**: Workflow diagram and phase descriptions
2. **Phase 1**: Initial LightGBM feature selection
3. **Phase 2**: Binary classification with XAI analysis
4. **Phase 3**: Multi-class LightGBM preprocessing
5. **Phase 4**: Multi-class classification with class-specific XAI

## 🔧 Configuration

### Evaluation Settings (`evaluation_config.yaml`)

```yaml
evaluation_settings:
  sample_size: 200                  # Samples for evaluation
  random_seed: 42                   # Reproducibility
  parallel_processing: true         # Enable parallelization
  max_workers: 4                    # Worker threads

dimension_weights:
  faithfulness: 0.40                # Faithfulness weight
  robustness: 0.35                  # Robustness weight
  complexity: 0.25                  # Complexity weight

faithfulness_config:
  deletion_steps: 10                # Steps for deletion test
  perturbation_method: "zero"       # Masking method

robustness_config:
  perturbation_levels: [0.01, 0.02, 0.05]
  bootstrap_samples: 10

complexity_config:
  sparsity_weight: 0.6
  effective_complexity_weight: 0.4
```

### Environment Variables

```bash
# LLM Configuration
export DEEPSEEK_API_KEY='your_key'
export DEEPSEEK_TIMEOUT=60
export DEEPSEEK_MAX_RETRIES=4
export DEEPSEEK_MAX_SAMPLES=3

# Evaluation Configuration
export XAI_EVAL_SAMPLE_SIZE=200
export XAI_EVAL_STRATIFIED=1

# Workflow Control
export GENERATE_LLM_EXPLANATIONS=1
```

## 📖 Supported Datasets

### 1. 5G-NIDD (5G Network Intrusion Detection Dataset)
- **Size**: 1.2M+ samples
- **Features**: Network traffic characteristics
- **Classes**: Benign, Multiple attack types
- **Preprocessing**: Feature cleaning, one-hot encoding

### 2. 5GAD (5G Attack Dataset)
- **Features**: Packet-level network features
- **Classes**: Normal, Attack
- **Preprocessing**: TCP flags encoding, IP removal

### 3. 5GC_PFCP (5G Core PFCP Dataset)
- **Features**: PFCP protocol characteristics
- **Classes**: Normal, Various attacks
- **Preprocessing**: Label mapping

## 🎓 Use Cases

### 1. Security Operations Center (SOC)
- Deploy models with verified XAI quality
- Use LLM explanations for incident response
- Monitor explanation quality over time

### 2. Research & Development
- Benchmark XAI methods across models
- Optimize explanation quality metrics
- Develop new XAI techniques

### 3. Compliance & Audit
- Generate auditable model explanations
- Demonstrate model interpretability
- Meet regulatory requirements

## 🔬 Advanced Usage

### Custom Model Integration

```python
from workflow import train_and_evaluate_model

# Train custom model
custom_model = YourCustomModel()
results = train_and_evaluate_model(
    custom_model, 
    X_train, y_train, 
    X_test, y_test,
    model_name="CustomModel",
    task_type="binary"
)
```

### Batch Evaluation

```python
from evaluation_toolkit import SimplifiedXAIEvaluator

evaluator = SimplifiedXAIEvaluator(config={
    'sample_size': 500,
    'bootstrap_samples': 20
})

# Evaluate multiple models
for model_name in ['DecisionTree', 'RandomForest', 'XGBoost']:
    model_dir = f"results/binary/{model_name}"
    # ... evaluation logic
```

### Custom LLM Prompts

Edit prompts in `llm_explanation_module.py`:

```python
system_prompt = """You are a cybersecurity expert...
[Your custom instructions here]
"""
```

## 📈 Performance Metrics

Typical performance on 5GAD dataset:

| Model | Accuracy | F1-Score | Faithfulness | Robustness | Complexity |
|-------|----------|----------|--------------|------------|------------|
| RandomForest | 99.8% | 0.998 | 0.89 | 0.92 | 0.85 |
| XGBoost | 99.7% | 0.997 | 0.87 | 0.90 | 0.83 |
| DNN | 99.5% | 0.995 | 0.85 | 0.88 | 0.80 |
| DecisionTree | 98.9% | 0.989 | 0.91 | 0.86 | 0.88 |
| MLP | 99.4% | 0.994 | 0.84 | 0.87 | 0.79 |

## 🏆 Best Practices

### 1. Workflow Optimization
- Use stratified sampling for evaluation (preserves class distribution)
- Enable parallel processing for large datasets
- Cache LLM explanations to reduce API costs

### 2. Quality Assessment
- Start with default configuration, then fine-tune
- Evaluate on representative samples (200+ recommended)
- Monitor all three dimensions (faithfulness, robustness, complexity)

### 3. LLM Usage
- Set appropriate API timeout for large models
- Use caching to avoid redundant API calls
- Review generated explanations for accuracy

### 4. Production Deployment
- Select models with overall score > 0.7
- Validate explanations with domain experts
- Implement continuous monitoring of XAI quality

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
black *.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- **Documentation**: [Project Wiki](https://github.com/your-repo/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@xai-framework.com

## 🙏 Acknowledgments

This framework is built upon research and tools from:

- **SHAP**: [Lundberg & Lee, NeurIPS 2017](https://github.com/slundberg/shap)
- **LIME**: [Ribeiro et al., KDD 2016](https://github.com/marcotcr/lime)
- **Deepseek**: [Deepseek Platform](https://platform.deepseek.com/)
- **XAI Research**: "Navigating the Maze of Explainable AI" (NeurIPS 2024)

Special thanks to the open-source community for their invaluable contributions.


## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{xai_framework_2025,
  title = {XAI Framework for Network Intrusion Detection Systems},
  author = {Xinyu Gong},
  year = {2025},
  url = {https://github.com/your-repo/XAI-Framework}
}
```

## 🔗 Related Projects

- [SHAP](https://github.com/slundberg/shap) - Unified framework for model interpretation
- [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations
- [XAI Benchmark](https://github.com/salesforce/OmniXAI) - Comprehensive XAI library

## Systematic Performance Validation of Retrieval Strategies in XAI-RAG for 5G Threat Detection - Thesis RAG Extension

The Retrieval-Augmented Generation (RAG) experimental pipeline developed for the thesis is located in the `RAG/` directory.

It evaluates retrieval strategies for cybersecurity knowledge augmentation using:
- No RAG
- BM25 retrieval
- Vector-based retrieval

See the dedicated documentation:
[RAG Experimental Setup](./RAG/README.md)

---

**⭐ If you find this project useful, please give it a star!**

**💬 Questions? Open an issue or discussion - we're here to help!**

