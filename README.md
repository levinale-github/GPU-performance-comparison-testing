# GPU Benchmarking & Performance Testing Suite

Welcome to the **GPU Benchmarking & Performance Testing** repository! This project is a community-driven effort to measure and compare the performance of various graphics cards and GPU accelerators. Whether you're a newcomer trying to decide on your first GPU, or a seasoned pro benchmarking the latest hardware, this repo has something for you.

## Introduction
**What is this repository?**  
It's a one-stop collection of GPU benchmark tests and results. We provide easy-to-run scripts that test different aspects of GPU performance – from memory bandwidth and compute power to gaming and machine learning workloads. We also aggregate results submitted by the community so you can compare how different GPUs perform.

**Who is it for?**  
- *Curious about GPUs:* If you're not sure which graphics card to buy or want to understand the differences between models, you can use our guides and results to inform your decision.  
- *Hardware enthusiasts and professionals:* If you own a GPU (or several!), you can run our test suite to measure its performance. You can even contribute your results to our public database and see how your setup stacks up against others.

## Getting Started

### Prerequisites
- **Python 3.x** (for running the benchmark scripts and tools)
- **GPU Drivers**: Ensure you have the proper drivers installed for your GPU. For NVIDIA, install the CUDA drivers; for AMD, install Radeon drivers or ROCm (on Linux). For integrated GPUs (Intel/AMD APUs), make sure the graphics driver is up to date.
- Some benchmarks use additional tools:
  - *OpenCL:* Many tests use OpenCL for cross-platform compatibility. Install an OpenCL runtime for your GPU (e.g., Intel OpenCL drivers for iGPUs, AMD APP SDK for AMD GPUs if not already in driver).  
  - *Optional:* If on Linux, you can install **glmark2** for the graphics test via your package manager. On Windows, our script will download a pre-built GLMark2 or use a simple DirectX sample.
  - *Python libraries:* If any Python benchmarks are included (e.g., for ML), install the requirements with `pip install -r requirements.txt` (we will include a requirements file as needed).

### Running the Benchmarks
To run the full benchmark suite, clone this repository and execute the main script:
```bash
git clone https://github.com/YourUsername/gpu-benchmarks.git  
cd gpu-benchmarks  
python run_benchmarks.py
```

This will sequentially run all the tests in the suite. **Please be patient**, as some tests (especially rendering or deep learning tasks) may take a few minutes. When it completes, you will see a summary of results in your terminal, and a results file (e.g., `results/myGPU_results.json`) will be generated.

If you prefer to run individual tests (for example, just the memory bandwidth test), you can run the specific script. For instance:

```bash
python benchmarks/memory/mem_bandwidth.py
```

(Above is just an example; check the `benchmarks/` folder for the actual script names and usage instructions for each test.)

### Sample Output

After a successful run, your results file might look like this (in JSON format for example):

```json
{
  "GPU": "NVIDIA GeForce GTX 1080 Ti",
  "Driver": "GeForce 472.12 (Windows 10)",
  "Memory_Bandwidth": "484.5 GB/s",
  "Single-Precision_GFLOPS": 11200,
  "Double-Precision_GFLOPS": 350,
  "GLMark2_Score": 7500,
  "ResNet50_Inference_ips": 225,
  "Timestamp": "2025-04-10T21:00:00Z"
}
```

*(The above is just illustrative – actual fields and tests may differ.)*

You will also have a detailed log file (e.g., `myGPU_run.log`) capturing the console output of each test. It's a good idea to review it to ensure all tests ran correctly (no errors, GPU was utilized, etc.).

## Viewing and Comparing Results

All contributed results are visible on our **Public Dashboard**: [**GPU Benchmark Dashboard**](https://levinale-github.github.io/GPU-performance-comparison-testing/). There, you can interactively compare different GPUs on various metrics. For quick reference, we also maintain a **Results Summary** table in the results file in this repo.

Some things you can discover:

* Which GPU has the highest memory bandwidth, or the fastest AI inference speed.
* How integrated GPUs (like Intel Iris or AMD Radeon Vega in APUs) compare against discrete GPUs. (Spoiler: discrete GPUs are much faster, especially in memory-intensive tasks – system RAM is a bottleneck for integrated GPUs).
* The performance difference between GPU generations (e.g., RTX 20 series vs RTX 30 series).
* How a cloud GPU instance (like an AWS Tesla T4) compares to a consumer desktop GPU you might have at home.

## Contributing Your Results

**We welcome contributions!** If you've run the benchmarks on your GPU, you can submit your results to be included in our database. Here's how:

1. **Run the benchmarks** using the instructions above. Ensure you get a results file (JSON) and save the log file.
2. **Fork this repository** (click the "Fork" button in GitHub).
3. **Add your results** in the `results/submissions/` folder. You can name the file descriptively, for example: `2025-04-10_NVIDIA_GTX1080Ti.json`. 
4. **Privacy Note:** Log files may contain personally identifiable information (PII) such as usernames in file paths. If you choose to include your log file, please review it and redact any personal information. Alternatively, you can submit just your results JSON file without the log. See CONTRIBUTING.md for detailed guidance on handling PII.
5. **Open a Pull Request (PR)** from your fork to this repo's `main` branch. In the PR description, mention what GPU and driver the results are for, and any notes (if, say, you did something special like overclocking).
6. A maintainer will review your submission. They will check the data format and may cross-verify with your log. If everything looks good, we'll merge it in! After merge, the dashboard will automatically update to include your GPU's results.

Please see CONTRIBUTING.md for more details on the process and data format. We want to ensure all results are **credible and reproducible**, so we might ask for clarifications.

## Choosing the Right GPU

If you are new to GPUs and trying to make sense of the numbers: we have you covered! We provide a brief guide here, and further resources:

* **Understand the Metrics:** Higher memory bandwidth means the GPU can feed data to its processors faster, which helps in data-heavy tasks. More FLOPS means more raw compute power. Scores in graphics tests correlate with gaming performance. No single number tells the whole story, so consider the workload you care about most.

* **Use Our Dashboard:** Filter by the type of workload that matters to you. For example, if you do a lot of 3D rendering, see which GPUs top the Blender or GLMark2 charts. If you do machine learning, look at the ResNet inference results or tensor core tests.

* **Don't Have a GPU Yet?** If you're looking to purchase a GPU, you can compare features and prices across various models at [BrainyBuyer's Graphics Cards category](https://www.brainybuyer.com/categories/284822-1/graphics-cards). This can help you find the best value for your specific needs and budget constraints.

* **Community Help:** Feel free to open a discussion or issue in this repo if you want advice. The community of contributors might share insights (e.g., if a certain GPU has known driver issues affecting performance, or if a new GPU release is around the corner that's worth waiting for).

*By combining specs comparison and real benchmark data, you'll be well-equipped to choose a GPU that suits your needs and budget.*

## Project Dashboard and Website

All the data collected is aggregated and shown on our [dashboard website](https://levinale-github.github.io/GPU-performance-comparison-testing/). The dashboard automatically reads and processes all JSON files in the `results/submissions/` directory, so there's no need to maintain a separate aggregated results file. In addition, we may post occasional analysis or summaries in the repository Wiki or a `/docs` article (for those who prefer reading a report).

The dashboard is hosted using GitHub Pages. If you prefer offline access, you can download any of the JSON files from the `results/submissions/` directory and use them in your own analysis tools.

## Roadmap

This project is under active development. Here's a quick overview of upcoming features (see the Design Document in this repo for more details):

* **More Benchmarks:** We plan to add more tests, such as stress tests, ray-tracing performance, and additional game-like workloads.
* **Automated Testing:** We are exploring using continuous integration to run a subset of benchmarks on reference hardware for each new commit (to catch any issues).
* **Enhanced Visualization:** Filtering by GPU type (desktop vs laptop vs cloud), grouping results by driver versions, etc., on the dashboard.
* **User Requests:** If you have ideas or want a particular benchmark included, let us know! This is a community project.

## License

This repository is released under the MIT License (see LICENSE file). All benchmark code in `benchmarks/` is open-source. Note that some integrated tools may have their own licenses (for example, glmark2 is under GPL, etc.); by using them, you agree to their terms as well. Results contributed are assumed to be willingly donated to the public domain for the sake of open data sharing.

## Acknowledgments

Thanks to all the open-source projects that make these benchmarks possible. This includes tools like Phoronix Test Suite (for inspiration and possible integration), clpeak, glmark2, and many others. We also thank the early contributors who provided valuable feedback and test data.

Let's benchmark some GPUs and learn together! Happy testing 😀 