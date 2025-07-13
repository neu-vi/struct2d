## Struct2D Prompting on VSI-Bench Subset

### 🚀 Installation

Make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 🔽 Download Preprocessed Dataset

We provide a subset of the **VSI-Bench** dataset with all necessary preprocessing [on Hugging Face](https://huggingface.co/datasets/fangruiz/struct2d/blob/main/subset_eval_vsibench.zip). This subset is ready for evaluation with ground truth bounding box annotations.  
Please download and extract the contents into:

```bash
./zero-shot-prompting/subset_eval_vsibench
```

Additionally, we release the full pipeline used to:
- Generate BEV (Bird’s Eye View) images
- Project object marks onto those images
- 3D bounding boxes results from [Mask3D](https://github.com/JonasSchult/Mask3D)

These tools support both training data generation and evaluation on the full dataset.

### ✅ Evaluation on Subset

First, make sure your OpenAI API key is available in the environment:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Then run the evaluation on a selected question type using the following command:

```bash
python run_eval.py \
  --config ./config.yaml \
  --mode struct2d \
  --model-version o3 \
  --subset-id-path ./subset_ids/rel_direction.json \ # change this to other JSON files under 'subset_ids' for different question types
  --log-dir ./logs \
  --num-votes 5 \
  --num-workers 32 
```
You can evaluate different question types by modifying the `--subset-id-path`. All available subsets can be found under the `subset_ids/` directory.