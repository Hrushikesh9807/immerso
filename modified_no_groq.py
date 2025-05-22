

import cv2
import os
import random
import numpy as np
from PIL import Image
import torch
import json
import re
import time
from tqdm import tqdm

# --- Configuration ---
VIDEO_PATH = "/home/ldvgpu/anirud/projects/immerso/output_clips/clip_4.mp4"
FRAMES_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results_bajirao/4v1/frames_base_dir"
RESULTS_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results_bajirao/4v1/pipeline_results"
os.makedirs(FRAMES_BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_BASE_OUTPUT_DIR, exist_ok=True)

HF_TOKEN = "" # Keep your token secure

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf" # Using 7B as specified for all LLaVA tasks
CLIP_DURATION_SECONDS = 2

SAMPLING_METHODS_CONFIG = [
    {
        "name": "optical_flow",
        "frame_sampling_rate": 5,
        "enabled": True
    },
    {
        "name": "image_embedding",
        "frame_sampling_rate": 10,
        "enabled": True
    },
    {
        "name": "caption_embedding", # This method can generate many captions
        "frame_sampling_rate": 15,
        "enabled": True
    }
]

LLAVA_BATCH_SIZE = 1 # For Stage 2 image captioning
LLAVA_MAX_NEW_TOKENS = 150 # For Stage 2 image captioning

# --- Configuration for LLaVA in Stage 3 (Summarization & Story Elaboration) ---
# These values are estimates for LLaVA 1.5 7B (Vicuna 7B base, ~4096 context). Adjust as needed.
# LLaVA's effective context for text-only might be close to its underlying LLM's context.
LLAVA_EFFECTIVE_CONTEXT_WINDOW = 3800 # Conservative estimate for text processing after accounting for prompt structure
LLAVA_STAGE3_TEMPERATURE = 0.7
# Chunking LLaVA captions from Stage 2 if they are too long for a single Stage 3 LLaVA prompt
LLAVA_STAGE3_MAX_INPUT_TOKENS_PER_CHUNK_FOR_SUMMARIZATION = 3000 # Max tokens of concatenated Stage 2 captions per chunk
LLAVA_STAGE3_MAX_NEW_TOKENS_CHUNK_SUMMARY = 250          # Max output tokens for summary of each caption chunk by LLaVA

# For final structured summary (Background, Characters, Story) using LLaVA
LLAVA_STAGE3_MAX_INPUT_TOKENS_FINAL_SUMMARY = 3000       # Max tokens for aggregated chunk summaries fed to LLaVA
LLAVA_STAGE3_MAX_NEW_TOKENS_FINAL_SUMMARY = 700          # Max output tokens for LLaVA's final structured summary

# For story elaboration using LLaVA
LLAVA_STAGE3_MAX_INPUT_TOKENS_STORY_ELABORATION = 1000   # Max tokens for the story segment fed to LLaVA
LLAVA_STAGE3_MAX_NEW_TOKENS_STORY_ELABORATION = 700      # Max output tokens for LLaVA's elaborated story


# --- Helper Functions ---
def create_dummy_video_if_not_exists(video_path, min_duration_secs=35):
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}. Creating a dummy video.")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        if not out.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            return
        num_frames = int(fps * (min_duration_secs + 5))
        for i in range(num_frames):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        print(f"Created dummy video ({num_frames/fps:.1f}s): {video_path}")
    else:
        print(f"Using existing video: {video_path}")

create_dummy_video_if_not_exists(VIDEO_PATH, CLIP_DURATION_SECONDS * (len(SAMPLING_METHODS_CONFIG) * 3 + 10))

# Initialize model placeholders to None
img_processor, img_model = None, None
caption_processor, caption_model = None, None
text_embedding_model = None
# LLaVA models will be loaded/unloaded per stage/method iteration

# Import heavy libraries here
from transformers import (
    AutoProcessor, AutoModel, AutoModelForCausalLM,
    LlavaForConditionalGeneration, LlavaProcessor
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print(f"Using device for pipeline: {DEVICE}")

def get_image_embedding(frame_rgb_pil, current_img_processor, current_img_model):
    if current_img_processor is None or current_img_model is None:
        raise RuntimeError("Image embedding model/processor not loaded for get_image_embedding.")
    with torch.no_grad():
        inputs = current_img_processor(images=frame_rgb_pil, return_tensors="pt").to(DEVICE)
        outputs = current_img_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze().cpu().numpy()

def get_caption_and_embedding(frame_rgb_pil, current_cap_processor, current_cap_model, current_text_embed_model):
    if current_cap_processor is None or current_cap_model is None or current_text_embed_model is None:
        raise RuntimeError("Captioning or text embedding model/processor not loaded for get_caption_and_embedding.")
    with torch.no_grad():
        inputs = current_cap_processor(images=frame_rgb_pil, return_tensors="pt").to(DEVICE)
        pixel_values = inputs.pixel_values
        generated_ids = current_cap_model.generate(pixel_values=pixel_values, max_length=50)
        caption = current_cap_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        caption_embedding = current_text_embed_model.encode(caption)
        return caption, caption_embedding

def split_and_sample_advanced(
    video_path,
    output_dir_for_method,
    clip_duration=5,
    method='optical_flow',
    frame_sampling_rate=1,
    current_img_processor=None, current_img_model=None,
    current_caption_processor=None, current_caption_model=None,
    current_text_embedding_model=None
):
    os.makedirs(output_dir_for_method, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        print(f"Error: Video FPS ({fps}) or total frames ({total_frames}) is zero. Cannot process {video_path}")
        cap.release()
        return 0

    duration = total_frames / fps
    clip_frame_count = int(clip_duration * fps)

    if clip_frame_count == 0:
        print(f"Error: Clip duration ({clip_duration}s) results in zero frames per clip for FPS ({fps}).")
        cap.release()
        return 0

    num_clips = int(total_frames // clip_frame_count)
    if num_clips == 0:
        print(f"Warning: Video duration ({duration:.2f}s) is less than clip duration ({clip_duration}s).")
        if total_frames > 0 :
            num_clips = 1
            print(f"Processing the video as a single partial clip (Frames 0 to {total_frames -1}).")
        else:
            cap.release()
            return 0

    print(f"\n--- Sampling for method: {method} ---")
    print(f"Video: {video_path}")
    print(f"Total duration: {duration:.2f} sec, FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"Splitting into {num_clips} clips of ~{clip_duration} seconds ({clip_frame_count} frames) each.")
    print(f"Sampling every {frame_sampling_rate} frame(s) within each clip for consideration.")

    if method == 'image_embedding' and (current_img_processor is None or current_img_model is None):
        print("Error: Image embedding model not loaded. Skipping sampling.")
        cap.release()
        return 0
    if method == 'caption_embedding' and \
       (current_caption_processor is None or current_caption_model is None or current_text_embedding_model is None):
        print("Error: Captioning/text embedding models not loaded. Skipping sampling.")
        cap.release()
        return 0

    saved_frames_count = 0
    for clip_idx in range(num_clips):
        start_frame_idx = clip_idx * clip_frame_count
        end_frame_idx = min((clip_idx + 1) * clip_frame_count, total_frames)
        print(f"\nProcessing Clip {clip_idx + 1}/{num_clips} (Frames {start_frame_idx} to {end_frame_idx - 1})...")

        clip_frames_data_for_selection = []
        processed_frame_info = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        prev_gray_for_optical_flow = None
        frames_read_in_clip = 0

        for current_frame_num_in_video in range(start_frame_idx, end_frame_idx):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {current_frame_num_in_video}. Skipping rest of clip.")
                break
            frames_read_in_clip += 1
            is_first_frame_to_sample = (frames_read_in_clip == 1)
            is_nth_frame_to_sample = ((frames_read_in_clip - 1) % frame_sampling_rate == 0)

            if not (is_first_frame_to_sample or is_nth_frame_to_sample):
                if method == 'optical_flow':
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_gray_for_optical_flow = current_gray
                del frame
                continue

            print(f"  Considering frame {current_frame_num_in_video} for {method}...")
            clip_frames_data_for_selection.append((current_frame_num_in_video, frame.copy()))

            if method == 'optical_flow':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray_for_optical_flow is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray_for_optical_flow, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    avg_magnitude = np.mean(magnitude)
                    processed_frame_info.append({'frame_num': current_frame_num_in_video, 'metric': avg_magnitude})
                prev_gray_for_optical_flow = gray
            elif method in ['image_embedding', 'caption_embedding']:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                try:
                    if method == 'image_embedding':
                        embedding = get_image_embedding(frame_pil, current_img_processor, current_img_model)
                        processed_frame_info.append({'frame_num': current_frame_num_in_video, 'embedding': embedding})
                    elif method == 'caption_embedding':
                        caption, embedding = get_caption_and_embedding(frame_pil, current_caption_processor, current_caption_model, current_text_embedding_model)
                        processed_frame_info.append({'frame_num': current_frame_num_in_video, 'embedding': embedding, 'caption': caption})
                except Exception as e:
                    print(f"  Warning: Failed {method} for frame {current_frame_num_in_video}: {e}")
            del frame

        selected_frame_video_idx = -1
        if not processed_frame_info:
            if clip_frames_data_for_selection:
                selected_frame_video_idx = clip_frames_data_for_selection[0][0]
            else: continue
        else:
            if method == 'optical_flow':
                if processed_frame_info:
                    processed_frame_info.sort(key=lambda item: item['metric'], reverse=True)
                    selected_frame_video_idx = processed_frame_info[0]['frame_num']
                elif clip_frames_data_for_selection: selected_frame_video_idx = clip_frames_data_for_selection[0][0]
            elif method in ['image_embedding', 'caption_embedding']:
                if len(processed_frame_info) > 0:
                    embeddings = np.array([item['embedding'] for item in processed_frame_info])
                    if embeddings.ndim == 1: embeddings = embeddings[np.newaxis, :]
                    if embeddings.shape[0] > 0:
                        mean_embedding = np.mean(embeddings, axis=0, keepdims=True)
                        sims = cosine_similarity(embeddings, mean_embedding).flatten()
                        best_match_local_idx = np.argmax(sims)
                        selected_item = processed_frame_info[best_match_local_idx]
                        selected_frame_video_idx = selected_item['frame_num']
                    elif clip_frames_data_for_selection: selected_frame_video_idx = clip_frames_data_for_selection[0][0]
                elif clip_frames_data_for_selection: selected_frame_video_idx = clip_frames_data_for_selection[0][0]
            else:
                 if clip_frames_data_for_selection:
                    selected_frame_video_idx = clip_frames_data_for_selection[random.randint(0, len(clip_frames_data_for_selection)-1)][0]

        if selected_frame_video_idx != -1:
            found_frame_data = next((f_data for f_num, f_data in clip_frames_data_for_selection if f_num == selected_frame_video_idx), None)
            if found_frame_data is not None:
                frame_filename = os.path.join(output_dir_for_method, f"clip_{clip_idx+1}_{method}_frame_{selected_frame_video_idx}.jpg")
                cv2.imwrite(frame_filename, found_frame_data)
                saved_frames_count += 1
        del clip_frames_data_for_selection[:]; del processed_frame_info[:]
        # Clean up local vars
        for var_name in ['gray', 'current_gray', 'flow', 'frame_rgb', 'frame_pil', 'embedding', 'embeddings']:
            if var_name in locals(): del locals()[var_name]
    cap.release()
    print(f"--- Finished sampling for method: {method}. Total frames saved: {saved_frames_count} ---")
    return saved_frames_count

def extract_sections_xml(text):
    sections = {"background": "Not specified.", "characters": "Not specified.", "story": "Not specified."}
    if not text: return sections
    bg_match = re.search(r"<Background>(.*?)</Background>", text, re.IGNORECASE | re.DOTALL)
    if bg_match: sections["background"] = bg_match.group(1).strip()
    char_match = re.search(r"<Characters>(.*?)</Characters>", text, re.IGNORECASE | re.DOTALL)
    if char_match: sections["characters"] = char_match.group(1).strip()
    story_match = re.search(r"<Story>(.*?)</Story>", text, re.IGNORECASE | re.DOTALL)
    if story_match: sections["story"] = story_match.group(1).strip()
    return sections

def llava_text_generate(prompt_text, llava_proc, llava_mod, max_new_tokens, temperature):
    """ Helper for text-only generation with LLaVA """
    inputs = llava_proc(text=prompt_text, return_tensors="pt").to(llava_mod.device)
    # Ensure no pixel_values are accidentally passed from a previous image-based call
    if 'pixel_values' in inputs:
        del inputs['pixel_values']

    with torch.no_grad():
        outputs = llava_mod.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0) # do_sample True if temperature > 0
        )
    # Decode, removing the prompt part if LLaVA includes it (often does with USER/ASSISTANT)
    full_decoded_text = llava_proc.decode(outputs[0], skip_special_tokens=True)
    
    # Try to extract only the assistant's response
    if "ASSISTANT:" in full_decoded_text:
        generated_text = full_decoded_text.split("ASSISTANT:")[-1].strip()
    else: # Fallback if ASSISTANT: marker is not found
        generated_text = full_decoded_text 
        # Heuristic: if the prompt is echoed, remove it. This is tricky.
        # For now, we assume LLaVA with USER/ASSISTANT prompt correctly splits.
        # If not, further prompt engineering or output parsing is needed.
        # A simple check could be: if generated_text.startswith(prompt_text.split("USER: ")[1].split("\nASSISTANT:")[0]):
        #    generated_text = generated_text[len(prompt_text.split("USER: ")[1].split("\nASSISTANT:")[0]):].strip()
    return generated_text

# --- Main Processing Loop ---
for method_config in SAMPLING_METHODS_CONFIG:
    method_name = method_config["name"]
    frame_sampling_rate = method_config["frame_sampling_rate"]

    if not method_config["enabled"]:
        print(f"\nSkipping method: {method_name} as it's disabled in config.")
        continue

    print(f"\n\n================ PROCESSING METHOD: {method_name.upper()} ================")

    # Stage 0: Load models for sampling
    torch.cuda.empty_cache()
    temp_img_processor, temp_img_model = None, None
    temp_caption_processor, temp_caption_model, temp_text_embedding_model = None, None, None
    try:
        if method_name == 'image_embedding':
            print(f"Loading models for image_embedding...")
            temp_img_processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            temp_img_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE)
        elif method_name == 'caption_embedding':
            print(f"Loading models for caption_embedding...")
            temp_caption_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
            temp_caption_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(DEVICE)
            temp_text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    except Exception as e:
        print(f"ERROR: Could not load models for sampling method {method_name}: {e}")
        # Ensure these are None so split_and_sample_advanced skips if models are crucial
        temp_img_processor, temp_img_model = None, None
        temp_caption_processor, temp_caption_model, temp_text_embedding_model = None, None, None

    current_method_frames_dir = os.path.join(FRAMES_BASE_OUTPUT_DIR, method_name)
    print(f"Stage 1: Sampling frames for {method_name} into {current_method_frames_dir}")
    start_time_sampling = time.time()
    num_sampled_frames = split_and_sample_advanced(
        VIDEO_PATH, output_dir_for_method=current_method_frames_dir,
        clip_duration=CLIP_DURATION_SECONDS, method=method_name, frame_sampling_rate=frame_sampling_rate,
        current_img_processor=temp_img_processor, current_img_model=temp_img_model,
        current_caption_processor=temp_caption_processor, current_caption_model=temp_caption_model,
        current_text_embedding_model=temp_text_embedding_model
    )
    print(f"Frame sampling for {method_name} took: {time.time() - start_time_sampling:.2f}s. Sampled {num_sampled_frames} frames.")

    print(f"Unloading sampling models for method {method_name}...")
    if temp_img_model: del temp_img_model
    if temp_img_processor: del temp_img_processor
    if temp_caption_model: del temp_caption_model
    if temp_caption_processor: del temp_caption_processor
    if temp_text_embedding_model: del temp_text_embedding_model
    temp_img_processor, temp_img_model = None, None
    temp_caption_processor, temp_caption_model, temp_text_embedding_model = None, None, None
    if DEVICE == "cuda": torch.cuda.empty_cache()
    print("Sampling models unloaded.")

    if num_sampled_frames == 0:
        print(f"No frames sampled for {method_name}. Skipping LLaVA stages.")
        continue

    # --- Stage 2: LLaVA Captioning (Image to Text) ---
    torch.cuda.empty_cache()
    llava_captions_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"llava_captions_{method_name}.json")
    llava_results_stage2 = [] # Use a distinct name
    stage2_llava_processor, stage2_llava_model = None, None
    print(f"\nStage 2: Generating LLaVA captions for {method_name}")
    try:
        print(f"Loading LLaVA model for Stage 2: {LLAVA_MODEL_ID}")
        stage2_llava_processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID, trust_remote_code=True)
        stage2_llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", trust_remote_code=True
        )
        print("LLaVA model (Stage 2) loaded.")

        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = sorted([
            os.path.join(current_method_frames_dir, f)
            for f in os.listdir(current_method_frames_dir)
            if os.path.isfile(os.path.join(current_method_frames_dir, f)) and \
               any(f.lower().endswith(ext) for ext in valid_exts)
        ])

        if not image_files:
            print(f"No image files found in {current_method_frames_dir} for LLaVA Stage 2.")
        else:
            start_time_llava_s2 = time.time()
            for i in tqdm(range(0, len(image_files), LLAVA_BATCH_SIZE), desc=f"LLaVA S2 Captions ({method_name})"):
                batch_paths = image_files[i:i+LLAVA_BATCH_SIZE]
                batch_images_pil, batch_prompts = [], []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_images_pil.append(img)
                        # Standard LLaVA prompt for image captioning
                        batch_prompts.append(f"USER: <image>\nDescribe this image in detail.\nASSISTANT:")
                    except Exception as e:
                        print(f"⚠️ Failed to load {path} for LLaVA S2: {e}")
                if not batch_images_pil: continue

                try:
                    inputs = stage2_llava_processor(text=batch_prompts, images=batch_images_pil, return_tensors="pt", padding=True).to(stage2_llava_model.device)
                    with torch.no_grad():
                        outputs = stage2_llava_model.generate(**inputs, max_new_tokens=LLAVA_MAX_NEW_TOKENS, do_sample=False) # No sampling for direct caption
                    
                    for idx, out_ids in enumerate(outputs):
                        full_text = stage2_llava_processor.decode(out_ids, skip_special_tokens=True)
                        # Extract only assistant's response
                        caption_part = full_text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_text else full_text
                        llava_results_stage2.append({"id": os.path.basename(batch_paths[idx]), "caption": caption_part})

                except Exception as e:
                    print(f"⚠️ Error during LLaVA S2 inference for a batch: {e}")
                    for path in batch_paths: # Add error placeholders if batch fails
                         llava_results_stage2.append({"id": os.path.basename(path), "caption": "Error during S2 caption generation."})


            with open(llava_captions_output_json, "w") as f: json.dump(llava_results_stage2, f, indent=2)
            print(f"✅ Saved LLaVA S2 captions for {len(llava_results_stage2)} images ({method_name}) to {llava_captions_output_json}")
            print(f"LLaVA S2 captioning for {method_name} took: {time.time() - start_time_llava_s2:.2f}s")
    except Exception as e:
        print(f"ERROR: Could not load or run LLaVA S2 model: {e}")
    finally:
        print("Unloading LLaVA S2 model...")
        if stage2_llava_model: del stage2_llava_model
        if stage2_llava_processor: del stage2_llava_processor
        stage2_llava_model, stage2_llava_processor = None, None
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print("LLaVA S2 model unloaded.")


    # --- Stage 3: LLaVA for Summarization and Story Generation (Text to Text) ---
    torch.cuda.empty_cache()
    stage3_llava_processor, stage3_llava_model = None, None # Distinct from Stage 2

    if not llava_results_stage2: # Check results from Stage 2
        print(f"No LLaVA S2 captions available for {method_name}. Skipping LLaVA Stage 3.")
        continue

    print(f"\nStage 3: LLaVA processing (Summarization & Story) for {method_name}")
    try:
        print(f"Loading LLaVA model for Stage 3: {LLAVA_MODEL_ID}")
        # Load LLaVA again for Stage 3 tasks
        stage3_llava_processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID, trust_remote_code=True)
        stage3_llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map="auto", trust_remote_code=True
        )
        print("LLaVA model (Stage 3) loaded.")

        captions_for_stage3 = [item["caption"] for item in llava_results_stage2 if item["caption"] and not item["caption"].lower().startswith("error")]
        if not captions_for_stage3:
            print(f"No valid LLaVA S2 captions for Stage 3 input ({method_name}). Skipping.")
            raise Exception("No valid captions for Stage 3") # To trigger finally block for cleanup

        # --- 3.1: Chunk and Summarize Captions with LLaVA (if needed) ---
        full_captions_text = "\n".join([f"`{i+1}`. {cap}" for i, cap in enumerate(captions_for_stage3)])
        # Tokenize the full text to see if chunking is needed
        tokenized_full_captions = stage3_llava_processor.tokenizer.encode(full_captions_text)
        num_full_tokens = len(tokenized_full_captions)
        print(f"Total tokens in concatenated S2 captions for {method_name}: {num_full_tokens}")

        summarized_input_for_final_step = ""

        if num_full_tokens > LLAVA_STAGE3_MAX_INPUT_TOKENS_PER_CHUNK_FOR_SUMMARIZATION:
            print(f"Input too long ({num_full_tokens} tokens), chunking S2 captions for LLaVA summarization...")
            all_chunk_summaries_s3 = []
            
            # Simple text splitting by character count as a proxy for tokens (crude but fast)
            # A more robust way is to split by token counts using the tokenizer
            avg_chars_per_token = len(full_captions_text) / num_full_tokens if num_full_tokens > 0 else 10 # Estimate
            estimated_chars_per_chunk = int(LLAVA_STAGE3_MAX_INPUT_TOKENS_PER_CHUNK_FOR_SUMMARIZATION * avg_chars_per_token * 0.9) # 0.9 for safety

            text_chunks = []
            current_pos = 0
            while current_pos < len(full_captions_text):
                text_chunks.append(full_captions_text[current_pos : current_pos + estimated_chars_per_chunk])
                current_pos += estimated_chars_per_chunk
            
            print(f"Split into {len(text_chunks)} text chunks for LLaVA S3 chunk summarization.")

            for i, text_chunk in enumerate(tqdm(text_chunks, desc=f"LLaVA S3 Chunk Summaries ({method_name})")):
                chunk_summary_prompt = (
                    f"USER: Concisely summarize the key events, people, and settings from the following video frame captions. "
                    f"Focus on the narrative flow if possible:\n\n{text_chunk}\n\nASSISTANT:"
                )
                try:
                    # Truncate prompt if it's still too long after crude char splitting
                    # (tokenizer handles this if input_ids become too long for model)
                    input_ids = stage3_llava_processor.tokenizer.encode(chunk_summary_prompt, return_tensors="pt", truncation=True, max_length=LLAVA_EFFECTIVE_CONTEXT_WINDOW - LLAVA_STAGE3_MAX_NEW_TOKENS_CHUNK_SUMMARY).to(stage3_llava_model.device)

                    chunk_summary = llava_text_generate(
                        stage3_llava_processor.tokenizer.decode(input_ids[0]), # Pass the potentially truncated prompt
                        stage3_llava_processor, stage3_llava_model,
                        LLAVA_STAGE3_MAX_NEW_TOKENS_CHUNK_SUMMARY, LLAVA_STAGE3_TEMPERATURE
                    )
                    all_chunk_summaries_s3.append(chunk_summary)
                except Exception as e:
                    print(f"  ⚠️ Error summarizing LLaVA S3 chunk {i+1}: {e}")
                    all_chunk_summaries_s3.append(f"[Error summarizing S3 text chunk {i+1}]")
            summarized_input_for_final_step = "\n---\n".join(all_chunk_summaries_s3)
            print(f"Finished LLaVA S3 chunk summarization. {len(all_chunk_summaries_s3)} chunk summaries generated.")
        else:
            print("S2 captions short enough, using them directly for LLaVA S3 final summary.")
            summarized_input_for_final_step = full_captions_text

        if not summarized_input_for_final_step.strip():
            print(f"No text available after chunk summarization for {method_name}. Skipping further S3 steps.")
            raise Exception("No text after chunking for Stage 3")

        # --- 3.2: Generate Final Structured Summary with LLaVA ---
        prompt_final_summary_s3 = (
            f"USER: You are a helpful video analysis assistant. "
            f"The following text contains descriptions or summaries derived from video frames, sampled using the '{method_name}' method.\n"
            f"Based *only* on this provided text, provide an overall summary of the entire video sequence.\n"
            f"The overall summary must include:\n"
            f"1. The overall background or setting.\n"
            f"2. The main characters or subjects observed.\n"
            f"3. A continuous narrative or story of the actions and events in chronological order.\n\n"
            f"Please structure your answer *exactly* as follows, using these XML-like tags:\n"
            f"<Background>Details about the background and setting.</Background>\n"
            f"<Characters>Details about characters or main subjects.</Characters>\n"
            f"<Story>A continuous narrative of events based on the provided text.</Story>\n\n"
            f"Provided Text:\n---\n{summarized_input_for_final_step}\n---\n\nASSISTANT:"
        )
        llava_final_summary_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"llava_S3_structured_summary_{method_name}.json")
        start_time_s3_final_summary = time.time()
        
        input_ids_final = stage3_llava_processor.tokenizer.encode(prompt_final_summary_s3, return_tensors="pt", truncation=True, max_length=LLAVA_EFFECTIVE_CONTEXT_WINDOW - LLAVA_STAGE3_MAX_NEW_TOKENS_FINAL_SUMMARY).to(stage3_llava_model.device)

        final_summary_text_s3 = llava_text_generate(
            stage3_llava_processor.tokenizer.decode(input_ids_final[0]),
            stage3_llava_processor, stage3_llava_model,
            LLAVA_STAGE3_MAX_NEW_TOKENS_FINAL_SUMMARY, LLAVA_STAGE3_TEMPERATURE
        )

        with open(llava_final_summary_output_json, "w") as f: json.dump({f"llava_s3_structured_summary_{method_name}": final_summary_text_s3}, f, indent=2)
        print(f"✅ LLaVA S3 Structured Summary ({method_name}) saved to {llava_final_summary_output_json}")
        print(f"LLaVA S3 Structured Summary for {method_name} took: {time.time() - start_time_s3_final_summary:.2f}s")

        if not final_summary_text_s3:
            print(f"No summary text from LLaVA S3 Final Summary for {method_name}. Skipping story elaboration.")
            raise Exception("No text from LLaVA S3 final summary")

        parsed_sections_s3 = extract_sections_xml(final_summary_text_s3)
        # Save the parsed sections separately as well
        parsed_output_path = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"llava_S3_parsed_summary_{method_name}.json")
        with open(parsed_output_path, "w") as f: json.dump({"id": method_name, **parsed_sections_s3}, f, indent=2)
        print(f"✅ Parsed LLaVA S3 Summary ({method_name}) saved to {parsed_output_path}")


        story_segment_from_s3_summary = parsed_sections_s3.get("story", "No story segment found.")
        if not story_segment_from_s3_summary or story_segment_from_s3_summary.lower() in ["not specified.", "no story segment found."]:
            print(f"No valid story segment extracted from LLaVA S3 summary for {method_name}. Skipping story elaboration.")
        else:
            # --- 3.3: Elaborate the Story Segment with LLaVA ---
            prompt_story_elaboration_s3 = (
                f"USER: You are a skilled storyteller. "
                f"You are given a narrative segment that was extracted from a video analysis. "
                f"Your task is to elaborate on this narrative, making it more descriptive, engaging, and flow like a continuous story. "
                f"Maintain factual consistency with the provided narrative. Do not invent new core events not implied by it. "
                f"Focus on detailing the actions and painting a clearer picture.\n\n"
                f"Provided Narrative Segment:\n---\n{story_segment_from_s3_summary}\n---\n\nElaborated Continuous Story:\nASSISTANT:"
            )
            llava_final_story_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"llava_S3_elaborated_story_{method_name}.json")
            start_time_s3_elaboration = time.time()

            input_ids_elaboration = stage3_llava_processor.tokenizer.encode(prompt_story_elaboration_s3, return_tensors="pt", truncation=True, max_length=LLAVA_EFFECTIVE_CONTEXT_WINDOW - LLAVA_STAGE3_MAX_NEW_TOKENS_STORY_ELABORATION).to(stage3_llava_model.device)
            
            final_story_text_s3 = llava_text_generate(
                stage3_llava_processor.tokenizer.decode(input_ids_elaboration[0]),
                stage3_llava_processor, stage3_llava_model,
                LLAVA_STAGE3_MAX_NEW_TOKENS_STORY_ELABORATION, LLAVA_STAGE3_TEMPERATURE
            )
            with open(llava_final_story_output_json, "w") as f: json.dump({f"llava_s3_elaborated_story_{method_name}": final_story_text_s3}, f, indent=2)
            print(f"✅ LLaVA S3 Elaborated Story ({method_name}) saved to {llava_final_story_output_json}")
            print(f"LLaVA S3 Story Elaboration for {method_name} took: {time.time() - start_time_s3_elaboration:.2f}s")

    except Exception as e:
        print(f"ERROR during LLaVA Stage 3 processing for {method_name}: {e}")
    finally:
        print("Unloading LLaVA S3 model...")
        if stage3_llava_model: del stage3_llava_model
        if stage3_llava_processor: del stage3_llava_processor
        stage3_llava_model, stage3_llava_processor = None, None
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print("LLaVA S3 model unloaded.")

print("\n\n================ ALL METHODS PROCESSED ================")