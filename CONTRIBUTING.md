# Contributing to GPU Benchmarking & Performance Testing

Thank you for your interest in contributing to our GPU benchmarking project! This document outlines the guidelines for contributing both benchmark results and code improvements.

## Contributing Benchmark Results

Sharing your GPU benchmark results is the primary way to help this project grow. Here's how to do it:

### Step 1: Run the Benchmarks

Run the benchmark suite on your GPU by following the instructions in the README.md:

```bash
python run_benchmarks.py
```

This will generate a results file (JSON) and a log file.

### Step 2: Prepare Your Submission

1. Fork this repository to your GitHub account
2. Clone your fork to your local machine
3. Create a new branch for your submission:

```bash
git checkout -b add-results-[your-gpu-model]
```

4. Copy your results file to the `results/submissions/` directory with a descriptive name:

```
results/submissions/[yyyy-mm-dd]_[vendor]_[model].json
```

For example: `2023-08-15_NVIDIA_RTX4080.json`

5. Copy your log file to the `results/logs/` directory with a matching name:

```
results/logs/[yyyy-mm-dd]_[vendor]_[model].log
```

**IMPORTANT: Log files and Privacy**  
Log files may contain personally identifiable information (PII) such as usernames and file paths. Before submitting, you should:
- Review your log files and redact any personal information
- Replace usernames in file paths with generic placeholders (e.g., `/Users/username/` → `/Users/anonymous/`)
- Alternatively, you can provide only the JSON results file without the log file if privacy is a concern

We have added these files to .gitignore, but you should still be careful when submitting logs to ensure no PII is included.

### Step 3: Submit a Pull Request

1. Commit your changes:

```bash
git add results/submissions/[your-file].json results/logs/[your-file].log
git commit -m "Add benchmark results for [GPU model]"
git push origin add-results-[your-gpu-model]
```

2. Go to GitHub and create a pull request from your branch to our main repository.
3. In the PR description, please include:
   - Your GPU model and details (e.g., "NVIDIA RTX 3080 10GB")
   - Driver version used
   - Any special notes about your setup (e.g., overclocking, cooling solution)
   - Operating system

### Data Format Requirements

Your benchmark results JSON should follow this structure:

```json
{
  "metadata": {
    "GPU": "NVIDIA GeForce RTX 3080",
    "GPUVendor": "NVIDIA",
    "Driver": "GeForce 535.98",
    "VRAM": "10 GB",
    "OS": "Windows 11",
    "CPUModel": "AMD Ryzen 7 5800X",
    "RAMSize": "32 GB",
    "Contributor": "Your GitHub username (optional)",
    "Notes": "Stock settings, no overclocking",
    "Timestamp": "2023-08-15T14:30:00Z",
    "Cost": {
      "RetailPrice": 699,
      "Currency": "USD",
      "PurchaseDate": "2022-01-15",
      "CloudEquivalent": "AWS g4dn.xlarge",
      "CloudCostPerHour": 0.526
    }
  },
  "results": {
    "memory": {
      "Bandwidth": 760.33,
      "BandwidthUnit": "GB/s",
      "TransferPCIe": 14.2,
      "TransferPCIeUnit": "GB/s"
    },
    "compute": {
      "FP32": 29.77,
      "FP32Unit": "TFLOPS",
      "FP16": 59.53,
      "FP16Unit": "TFLOPS",
      "INT8": 119.06,
      "INT8Unit": "TOPS"
    },
    "graphics": {
      "GLMark2Score": 9850,
      "BlenderClassroomSample": 163.4,
      "BlenderClassroomSampleUnit": "seconds"
    },
    "ai": {
      "ResNet50Inference": 1253,
      "ResNet50InferenceUnit": "images/sec",
      "BERTInference": 76.3,
      "BERTInferenceUnit": "samples/sec"
    }
  }
}
```

Note: Fill in only the benchmarks you've run. If you couldn't run certain tests or they're not applicable to your GPU, you can omit them.

The "Cost" section is optional but very valuable for our cost comparison feature. If you include it, try to be as accurate as possible with your GPU's retail price and purchase date.

## Contributing Code

We also welcome code contributions to improve the benchmark suite itself!

### Types of Code Contributions

1. **New Benchmarks**: Adding a new test to the suite
2. **Improvements**: Enhancing existing benchmarks
3. **Bug Fixes**: Fixing issues with benchmarks or the runner
4. **Dashboard Improvements**: Enhancing visualization or data analysis
5. **Documentation**: Improving guides and explanations

### Code Contribution Process

1. **Open an Issue First**: Before writing code, open an issue describing your planned contribution. This allows discussion and feedback before you invest time coding.

2. **Fork and Clone**: Fork the repository and clone it locally.

3. **Create a Branch**: Create a branch for your feature or fix.

4. **Implement Your Changes**: Write your code, following these guidelines:
   - Include clear documentation
   - Write tests for new functionality
   - Follow Python PEP 8 style guidelines
   - Keep cross-platform compatibility in mind (Windows/Linux/macOS where possible)

5. **Test Your Changes**: Ensure your code works on your system and doesn't break existing functionality.

6. **Submit a Pull Request**: Push your branch and create a PR with a clear description of the changes and the issue it addresses.

### Guidelines for Adding New Benchmarks

If you're adding a new benchmark:

1. Create a new directory under the appropriate category (e.g., `benchmarks/ai/new_benchmark/`)
2. Include a README.md explaining what the benchmark measures and how it works
3. Ensure the benchmark:
   - Has clear output formats
   - Handles errors gracefully
   - Works with the main runner script
   - Can run on multiple platforms (where possible)
   - Has reasonable dependencies (prefer widely available libraries)

## Code of Conduct

- Be respectful and constructive in discussions
- Focus on data accuracy and reproducibility
- Credit original sources for benchmarks or techniques
- Respect all GPU vendors equally (this is a vendor-neutral project)

Thank you for contributing to make GPU benchmarking more transparent and accessible for everyone!

If you have questions about contributing, please open an issue labeled "question" and we'll help you get started. 