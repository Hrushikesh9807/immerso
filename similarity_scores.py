import nltk
import evaluate

# Download necessary NLTK resources
# punkt is needed for tokenization by many metrics
# wordnet and omw-1.4 are often needed by METEOR

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# === File paths ===
generated_caption_path = "/home/ldvgpu/anirud/projects/immerso/captions_compare/22/caption_embedding.txt"
gt_caption_path = "/home/ldvgpu/anirud/projects/immerso/captions_compare/22/gt.txt"

# === Load captions ===
with open(generated_caption_path, "r", encoding="utf-8") as f:
    generated_caption = f.read().strip()

with open(gt_caption_path, "r", encoding="utf-8") as f:
    gt_caption = f.read().strip()

# === Prepare predictions and references ===
# Most metrics expect a list of predictions and a list of lists of references
predictions = [generated_caption]
references = [[gt_caption]] # Note: BLEU and CIDEr typically expect a list of lists of references

# === Load evaluation metrics ===
print("Loading BLEU metric...")
bleu = evaluate.load("bleu")
print("Loading ROUGE metric...")
rouge = evaluate.load("rouge")
print("Loading METEOR metric...")
meteor = evaluate.load("meteor")
# If the above still gives issues for cider, it might be because of dependencies
# for the ptbtokenizer. You might need:
# pip install pycocoevalcap
# And then ensure the java path is set if ptbtokenizer needs it (less common now).
# Or try the specific cider implementation from the coco-caption scripts if evaluate's version has issues.

print("\nComputing metrics...")
# === Compute similarity metrics ===
# Note on references format:
# - BLEU: expects references to be a list of lists of strings (e.g., [[ref1a, ref1b], [ref2a, ref2b]])
#   For a single prediction, it would be [[ref1a, ref1b, ...]]
#   So for your case with one GT, predictions=[pred], references=[[gt]] is correct.
# - ROUGE: can handle references as list of strings or list of lists.
#   For your case, predictions=[pred], references=[gt] or predictions=[pred], references=[[gt]] should work.
# - METEOR: similar to ROUGE. predictions=[pred], references=[gt] or predictions=[pred], references=[[gt]]
# - CIDEr: expects references like BLEU, i.e., list of lists of strings. predictions=[pred], references=[[gt]]

results = {}

print("Calculating BLEU...")
results["BLEU"] = bleu.compute(predictions=predictions, references=references)

print("Calculating ROUGE...")
# For ROUGE, if you have only one reference string per prediction,
# you can pass references as a list of strings directly.
# However, list of lists also works.
results["ROUGE"] = rouge.compute(predictions=predictions, references=references)

print("Calculating METEOR...")
# For METEOR, similar to ROUGE for single reference.
results["METEOR"] = meteor.compute(predictions=predictions, references=references)


# === Print the results ===
print("\n--- Results ---")
for metric, score_dict in results.items():
    print(f"{metric}:")
    if isinstance(score_dict, dict):
        for sub_metric, score_value in score_dict.items():
            if isinstance(score_value, float):
                print(f"  {sub_metric}: {score_value:.4f}")
            else:
                print(f"  {sub_metric}: {score_value}")
    else:
        if isinstance(score_dict, float):
             print(f"  Score: {score_dict:.4f}") # METEOR and CIDEr return a single float
        else:
            print(f"  Score: {score_dict}")
    print("-" * 15)