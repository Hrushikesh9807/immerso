

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
FRAMES_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results_bajirao/4/frames_base_dir"
RESULTS_BASE_OUTPUT_DIR = "/home/ldvgpu/anirud/projects/immerso/results_bajirao/4/pipeline_results"
os.makedirs(FRAMES_BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_BASE_OUTPUT_DIR, exist_ok=True)

HF_TOKEN = "" # Keep your token secure
GROQ_API_KEY = "" # Keep your API key secure

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-13b-hf"
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
GROQ_MODEL_ID = "llama3-70b-8192"
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
        "name": "caption_embedding",
        "frame_sampling_rate": 15,
        "enabled": True
    }
]

LLAVA_BATCH_SIZE = 1 # Reduced batch size as a precaution for VRAM
LLAVA_MAX_NEW_TOKENS = 150
GROQ_TEMPERATURE = 0.7
GROQ_MAX_TOKENS_SUMMARY = 1024
GROQ_MAX_TOKENS_STORY = 1024

# --- Helper Functions ---

def create_dummy_video_if_not_exists(video_path, min_duration_secs=35): # Ensure dummy video is long enough
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}. Creating a dummy video.")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        # Ensure enough frames for at least one clip based on CLIP_DURATION_SECONDS
        # Also consider the minimum duration for the dummy video itself
        effective_min_duration = max(min_duration_secs, CLIP_DURATION_SECONDS * 1.5) # Ensure dummy is a bit longer than one clip
        num_frames = int(fps * effective_min_duration)

        if num_frames == 0 : # Safety for very short clip durations potentially leading to 0 frames
            num_frames = int(fps * 5) # Default to 5 seconds if calculation is off
            print(f"Warning: Calculated num_frames for dummy video was 0. Defaulting to {num_frames} frames (5s).")

        out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        if not out.isOpened():
            print(f"Error: Could not open video writer for {video_path}")
            return

        for i in range(num_frames):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        out.release()
        print(f"Created dummy video ({num_frames/fps:.1f}s): {video_path}")
    else:
        print(f"Using existing video: {video_path}")

create_dummy_video_if_not_exists(VIDEO_PATH, CLIP_DURATION_SECONDS)

# Initialize model placeholders to None
img_processor, img_model = None, None
caption_processor, caption_model = None, None
text_embedding_model = None
llava_processor, llava_model = None, None

# Import heavy libraries here
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaProcessor
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity


print(f"Using device for sampling models: {DEVICE}")

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
    # Pass currently loaded models to avoid relying on globals directly within this function scope
    # These can be None if the method doesn't require them
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
        print(f"Error: Clip duration ({clip_duration}s) results in zero frames per clip for FPS ({fps}). This may be due to a very short clip duration or low FPS. Attempting to set clip_frame_count to 1 if fps > 0.")
        if fps > 0:
            clip_frame_count = 1 # Process one frame at a time if clip duration is too short.
            print("Adjusted clip_frame_count to 1.")
        else: # Should have been caught by earlier fps check, but as a safeguard
            cap.release()
            return 0


    num_clips = int(total_frames // clip_frame_count)
    if num_clips == 0:
        print(f"Warning: Video duration ({duration:.2f}s) is less than clip duration ({clip_duration}s) or results in 0 clips. Processing the video as a single (potentially partial) clip if frames exist.")
        if total_frames > 0 :
            num_clips = 1
            # Adjust clip_frame_count to total_frames for this single pass
            clip_frame_count = total_frames
            print(f"Processing the video as a single clip covering all {total_frames} frames.")
        else:
            cap.release()
            return 0


    print(f"\n--- Sampling for method: {method} ---")
    print(f"Video: {video_path}")
    print(f"Total duration: {duration:.2f} sec, FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"Splitting into {num_clips} clips of ~{clip_duration} seconds ({clip_frame_count} frames) each.")
    print(f"Sampling every {frame_sampling_rate} frame(s) within each clip for consideration.")

    if method == 'image_embedding' and (current_img_processor is None or current_img_model is None):
        print("Error: Image embedding model not loaded for 'image_embedding' method. Skipping sampling.")
        cap.release()
        return 0
    if method == 'caption_embedding' and \
       (current_caption_processor is None or current_caption_model is None or current_text_embedding_model is None):
        print("Error: Captioning/text embedding models not loaded for 'caption_embedding' method. Skipping sampling.")
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
            
            # Corrected sampling logic:
            # The first frame of the segment being considered should always be processed for context (e.g., optical flow).
            # Then, subsequent frames are sampled based on frame_sampling_rate.
            # current_frame_num_in_clip is 0-indexed relative to the start of the current segment being processed.
            current_frame_num_in_clip = current_frame_num_in_video - start_frame_idx

            if not (current_frame_num_in_clip == 0 or (current_frame_num_in_clip % frame_sampling_rate == 0 and frame_sampling_rate > 0) ):
                if method == 'optical_flow':
                    current_gray_for_of_update = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Always update prev_gray_for_optical_flow to the immediate prior frame,
                    # regardless of whether this frame is sampled for OF calculation
                    prev_gray_for_optical_flow = current_gray_for_of_update
                del frame # Release memory
                continue


            print(f"  Considering frame {current_frame_num_in_video} for {method}...")
            clip_frames_data_for_selection.append((current_frame_num_in_video, frame.copy()))

            if method == 'optical_flow':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray_for_optical_flow is not None and current_frame_num_in_clip > 0 : # Ensure there is a previous frame
                    flow = cv2.calcOpticalFlowFarneback(prev_gray_for_optical_flow, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    avg_magnitude = np.mean(magnitude)
                    processed_frame_info.append({'frame_num': current_frame_num_in_video, 'metric': avg_magnitude})
                prev_gray_for_optical_flow = gray # Update for the next sampled frame

            elif method == 'image_embedding' or method == 'caption_embedding':
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
        if not processed_frame_info and method != 'optical_flow': # For embedding methods, if no info, pick first
            if clip_frames_data_for_selection:
                selected_frame_video_idx = clip_frames_data_for_selection[0][0]
                print(f"  No frames processed by method '{method}' in clip {clip_idx+1} or only one frame sampled. Selecting first physically sampled frame: {selected_frame_video_idx}.")
            else:
                print(f"  No frames sampled or processed for Clip {clip_idx + 1}.")
                continue
        elif not processed_frame_info and method == 'optical_flow': # For optical flow, if no info (e.g. <2 frames sampled), pick first
            if clip_frames_data_for_selection:
                selected_frame_video_idx = clip_frames_data_for_selection[0][0]
                print(f"  Optical Flow: Not enough frames to calculate flow or only one frame sampled in clip {clip_idx+1}. Selecting first physically sampled frame: {selected_frame_video_idx}.")
            else:
                print(f"  No frames sampled or processed for Clip {clip_idx + 1}.")
                continue
        else: # processed_frame_info has items
            if method == 'optical_flow':
                processed_frame_info.sort(key=lambda item: item['metric'], reverse=True)
                selected_frame_video_idx = processed_frame_info[0]['frame_num']
                print(f"  Optical Flow: Selected frame {selected_frame_video_idx} (Max Avg Mag: {processed_frame_info[0]['metric']:.2f})")

            elif method in ['image_embedding', 'caption_embedding']:
                embeddings = np.array([item['embedding'] for item in processed_frame_info])
                if embeddings.ndim == 1: embeddings = embeddings[np.newaxis, :]

                if embeddings.shape[0] > 0: # Should be true if processed_frame_info is not empty
                    mean_embedding = np.mean(embeddings, axis=0, keepdims=True)
                    sims = cosine_similarity(embeddings, mean_embedding).flatten()
                    best_match_local_idx = np.argmax(sims)
                    selected_item = processed_frame_info[best_match_local_idx]
                    selected_frame_video_idx = selected_item['frame_num']
                    similarity_score = sims[best_match_local_idx]
                    if method == 'caption_embedding':
                        print(f"  {method}: Selected frame {selected_frame_video_idx} ('{selected_item.get('caption','N/A')}', Sim to mean: {similarity_score:.4f})")
                    else:
                        print(f"  {method}: Selected frame {selected_frame_video_idx} (Sim to mean: {similarity_score:.4f})")
                # This else case should not be hit if processed_frame_info is not empty for these methods
                elif clip_frames_data_for_selection: # Fallback just in case
                    selected_frame_video_idx = clip_frames_data_for_selection[0][0]
                    print(f"  {method}: No embeddings processed despite having sampled frames. Selecting first physically sampled frame {selected_frame_video_idx}.")


        if selected_frame_video_idx != -1:
            found_frame_data = None
            for f_num, f_data in clip_frames_data_for_selection:
                if f_num == selected_frame_video_idx:
                    found_frame_data = f_data
                    break
            if found_frame_data is not None:
                frame_filename = os.path.join(output_dir_for_method, f"clip_{clip_idx+1}_{method}_frame_{selected_frame_video_idx}.jpg")
                cv2.imwrite(frame_filename, found_frame_data)
                print(f"  SAVED: {frame_filename}")
                saved_frames_count += 1
            else:
                print(f"  Error: Could not find frame data for selected index {selected_frame_video_idx}.")
        else:
            print(f"  No frame selected for clip {clip_idx+1}.")

        del clip_frames_data_for_selection[:]
        del processed_frame_info[:]
        if 'gray' in locals(): del gray
        if 'flow' in locals(): del flow
        if 'frame_rgb' in locals(): del frame_rgb
        if 'frame_pil' in locals(): del frame_pil
        if 'embedding' in locals(): del embedding
        if 'embeddings' in locals(): del embeddings
        if 'current_gray_for_of_update' in locals(): del current_gray_for_of_update


    cap.release()
    print(f"--- Finished sampling for method: {method}. Total frames saved: {saved_frames_count} ---")
    return saved_frames_count


print("\nInitializing Groq client...")
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client initialized.")
except Exception as e:
    print(f"ERROR: Could not initialize Groq client: {e}")
    groq_client = None

def extract_sections_xml(text):
    sections = {"background": "Not specified.", "characters": "Not specified.", "story": "Not specified."}
    if not text:
        return sections

    # Background
    bg_match = re.search(r"<Background>(.*?)</Background>", text, re.IGNORECASE | re.DOTALL)
    if bg_match:
        content = bg_match.group(1).strip()
        if content:  # Only update if there's actual content after stripping
            sections["background"] = content
        # If content is empty after strip, it remains "Not specified."

    # Characters
    char_match = re.search(r"<Characters>(.*?)</Characters>", text, re.IGNORECASE | re.DOTALL)
    if char_match:
        content = char_match.group(1).strip()
        if content:
            sections["characters"] = content
        # If content is empty, it remains "Not specified."

    # Story
    story_match = re.search(r"<Story>(.*?)</Story>", text, re.IGNORECASE | re.DOTALL)
    if story_match:
        content = story_match.group(1).strip()
        if content:
            sections["story"] = content
        # If content is empty, it remains "Not specified."
            
    return sections


# --- Main Processing Loop ---
for method_config in SAMPLING_METHODS_CONFIG:
    method_name = method_config["name"]
    frame_sampling_rate = method_config["frame_sampling_rate"]

    if not method_config["enabled"]:
        print(f"\nSkipping method: {method_name} as it's disabled in config.")
        continue

    print(f"\n\n================ PROCESSING METHOD: {method_name.upper()} ================") # Moved print statement here

    torch.cuda.empty_cache()
    temp_img_processor, temp_img_model = None, None
    temp_caption_processor, temp_caption_model, temp_text_embedding_model = None, None, None

    try:
        if method_name == 'image_embedding':
            print(f"Loading models for image_embedding...")
            temp_img_processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            temp_img_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE)
            print("Image embedding model (ViT) loaded for current method.")
        elif method_name == 'caption_embedding':
            print(f"Loading models for caption_embedding...")
            temp_caption_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
            temp_caption_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco").to(DEVICE)
            temp_text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
            print("Captioning (GIT) and text embedding models loaded for current method.")
    except Exception as e:
        print(f"ERROR: Could not load models for sampling method {method_name}: {e}")
        temp_img_processor, temp_img_model = None, None
        temp_caption_processor, temp_caption_model, temp_text_embedding_model = None, None, None


    current_method_frames_dir = os.path.join(FRAMES_BASE_OUTPUT_DIR, method_name)
    print(f"Stage 1: Sampling frames for {method_name} into {current_method_frames_dir}")
    start_time_sampling = time.time()
    num_sampled_frames = split_and_sample_advanced(
        VIDEO_PATH,
        output_dir_for_method=current_method_frames_dir,
        clip_duration=CLIP_DURATION_SECONDS,
        method=method_name,
        frame_sampling_rate=frame_sampling_rate,
        current_img_processor=temp_img_processor, current_img_model=temp_img_model,
        current_caption_processor=temp_caption_processor, current_caption_model=temp_caption_model,
        current_text_embedding_model=temp_text_embedding_model
    )
    print(f"Frame sampling for {method_name} took: {time.time() - start_time_sampling:.2f}s. Sampled {num_sampled_frames} frames.")

    torch.cuda.empty_cache()
    print(f"Unloading sampling models for method {method_name}...")
    if temp_img_model: del temp_img_model; temp_img_model = None
    if temp_img_processor: del temp_img_processor; temp_img_processor = None
    if temp_caption_model: del temp_caption_model; temp_caption_model = None
    if temp_caption_processor: del temp_caption_processor; temp_caption_processor = None
    if temp_text_embedding_model: # This is a SentenceTransformer object, may not need explicit del for GPU mem as much as PyTorch models
        del temp_text_embedding_model; temp_text_embedding_model = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("Sampling models unloaded and cache cleared.")


    if num_sampled_frames == 0:
        print(f"No frames sampled for {method_name}. Skipping LLaVA and Groq stages.")
        continue

    import torch # Re-import or ensure it's available if previously conditionally imported
    torch.cuda.empty_cache()

    llava_captions_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"llava_captions_{method_name}.json")
    llava_results = []

    print(f"\nStage 2: Generating LLaVA captions for {method_name}")
    current_llava_processor, current_llava_model = None, None # Use local variables for LLaVA models
    try:
        print(f"Loading LLaVA model: {LLAVA_MODEL_ID}")
        current_llava_processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID, trust_remote_code=True)
        current_llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        print("LLaVA model and processor loaded.")

        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = sorted([
            os.path.join(current_method_frames_dir, f)
            for f in os.listdir(current_method_frames_dir)
            if any(f.lower().endswith(ext) for ext in valid_exts)
        ])

        if not image_files:
            print(f"No image files found in {current_method_frames_dir} for LLaVA captioning.")
        else:
            start_time_llava = time.time()
            for i in tqdm(range(0, len(image_files), LLAVA_BATCH_SIZE), desc=f"LLaVA Captions ({method_name})"):
                batch_paths = image_files[i:i+LLAVA_BATCH_SIZE]
                batch_images, batch_prompts = [], []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert("RGB")
                        batch_images.append(img)
                        # Standard prompt format for LLaVA 1.5
                        batch_prompts.append(f"USER: <image>\nDescribe this image in detail.\nASSISTANT:")
                    except Exception as e:
                        print(f"⚠️ Failed to load {path} for LLaVA: {e}")
                if not batch_images: continue

                try:
                    inputs = current_llava_processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True).to(current_llava_model.device)
                    with torch.no_grad():
                        outputs = current_llava_model.generate(**inputs, max_new_tokens=LLAVA_MAX_NEW_TOKENS, do_sample=False) # do_sample=False for more deterministic output

                    decoded_captions = []
                    for out_ids in outputs: # Iterate through batch outputs
                        full_text = current_llava_processor.decode(out_ids, skip_special_tokens=True)
                        # Extract only the assistant's response
                        if "ASSISTANT:" in full_text:
                            caption_part = full_text.split("ASSISTANT:")[-1].strip()
                        else: # Fallback if the ASSISTANT: marker isn't there as expected
                            caption_part = full_text.strip() # Or handle as an error/unexpected format
                        decoded_captions.append(caption_part)


                except Exception as e:
                    print(f"⚠️ Error during LLaVA inference for a batch: {e}")
                    decoded_captions = ["Error during caption generation."] * len(batch_paths) # Assign error to all in batch

                for path, caption in zip(batch_paths, decoded_captions):
                    llava_results.append({"id": os.path.basename(path), "caption": caption.strip()})

            with open(llava_captions_output_json, "w") as f:
                json.dump(llava_results, f, indent=2)
            print(f"✅ Saved LLaVA captions for {len(llava_results)} images ({method_name}) to {llava_captions_output_json}")
            print(f"LLaVA captioning for {method_name} took: {time.time() - start_time_llava:.2f}s")

    except Exception as e:
        print(f"ERROR: Could not load or run LLaVA model: {e}")
    finally:
        print("Unloading LLaVA model...")
        if current_llava_model is not None : del current_llava_model
        if current_llava_processor is not None : del current_llava_processor
        current_llava_model, current_llava_processor = None, None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("LLaVA model unloaded and cache cleared.")


    torch.cuda.empty_cache()
    if groq_client is None:
        print(f"Groq client not initialized. Skipping Groq stages for {method_name}.")
        continue
    if not llava_results:
        print(f"No LLaVA captions available for {method_name}. Skipping Groq summarization.")
        continue

    print(f"\nStage 3: Groq processing for {method_name}")
    captions_for_groq = [item["caption"] for item in llava_results if item["caption"] and not item["caption"].lower().startswith("error")]
    if not captions_for_groq:
        print(f"No valid LLaVA captions for Groq input ({method_name}). Skipping.")
        continue

    formatted_captions = "\n".join([f"Frame {i+1} Caption: {cap}" for i, cap in enumerate(captions_for_groq)]) # Slightly clearer prefix

    prompt1 = f"""You are a helpful video analysis assistant.
The following are a sequence of detailed captions derived from video frames. These frames were sampled using the '{method_name}' method.
Your task is to analyze these captions and provide a structured summary.
Based *only* on the information present in these captions:
1. Describe the overall background, setting, or environment.
2. Identify and describe the main characters, people, or primary subjects observed.
3. Weave together a continuous narrative or story detailing the actions, events, and developments as they appear sequentially in the captions.

Please structure your answer *exactly* as follows, using these specific XML-like tags. Ensure content for each section is within its respective tags:
<Background>Concise description of the background and setting.</Background>
<Characters>Clear identification and description of characters or main subjects.</Characters>
<Story>A continuous narrative of events, in order, based on the captions provided.</Story>

Video Frame Captions:
---
{formatted_captions}
---

Structured Summary:
"""
    groq_summary1_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"groq_summary1_{method_name}.json")
    start_time_groq1 = time.time()
    summary1_text = None
    try:
        print(f"Sending request to Groq for Summary 1 ({method_name})...")
        response1 = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[{"role": "user", "content": prompt1}],
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS_SUMMARY,
        )
        summary1_text = response1.choices[0].message.content.strip()
        # Save the raw response for debugging
        with open(groq_summary1_output_json, "w") as f:
            json.dump({f"raw_summary_for_{method_name}": summary1_text}, f, indent=2)
        print(f"✅ Groq Summary 1 ({method_name}) raw response saved to {groq_summary1_output_json}")
    except Exception as e:
        print(f"⚠️ Error during Groq Summary 1 ({method_name}): {e}")
        # Save error if any
        with open(groq_summary1_output_json, "w") as f:
            json.dump({f"error_summary_for_{method_name}": str(e)}, f, indent=2)
    print(f"Groq Summary 1 for {method_name} took: {time.time() - start_time_groq1:.2f}s")

    if not summary1_text:
        print(f"No summary text received from Groq Summary 1 for {method_name}. Skipping further Groq processing.")
        continue

    structured_summary_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"groq_summary1_structured_{method_name}.json")
    parsed_sections = extract_sections_xml(summary1_text)
    structured_output_data = {"id": method_name, "raw_groq_summary": summary1_text, **parsed_sections} # Include raw summary for context
    with open(structured_summary_output_json, "w") as f:
        json.dump(structured_output_data, f, indent=2)
    print(f"✅ Structured Groq Summary 1 ({method_name}) parsed and saved to {structured_summary_output_json}")

    story_segment_from_summary1 = parsed_sections.get("story", "Not specified.") # Default to "Not specified." if key missing
    if not story_segment_from_summary1 or story_segment_from_summary1 == "Not specified.":
        print(f"No valid story segment extracted from Groq Summary 1 for {method_name} (Story: '{story_segment_from_summary1}'). Skipping final Groq story generation.")
        continue

    prompt2 = f"""You are an expert storyteller and video scriptwriter.
You have been provided with a narrative segment that was automatically extracted from a sequence of video frame captions. This narrative outlines the key actions and events observed in a video clip.
Your task is to take this narrative segment and expand it into a more descriptive, engaging, and flowing story.
Enrich the descriptions, detail the actions, and make the sequence of events vivid and clear, as if narrating a short scene.
It is crucial to maintain factual consistency with the provided narrative segment. Do not introduce new core events, characters, or elements that are not implied by the original segment. Your goal is to elaborate, not invent.

Provided Narrative Segment:
---
{story_segment_from_summary1}
---

Elaborated Continuous Story:
"""
    groq_final_story_output_json = os.path.join(RESULTS_BASE_OUTPUT_DIR, f"groq_final_story_{method_name}.json")
    start_time_groq2 = time.time()
    try:
        print(f"Sending request to Groq for Final Story ({method_name})...")
        response2 = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[{"role": "user", "content": prompt2}],
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS_STORY,
        )
        final_story_text = response2.choices[0].message.content.strip()
        with open(groq_final_story_output_json, "w") as f:
            json.dump({f"final_story_for_{method_name}": final_story_text, "source_story_segment": story_segment_from_summary1}, f, indent=2)
        print(f"✅ Groq Final Story ({method_name}) saved to {groq_final_story_output_json}")
    except Exception as e:
        print(f"⚠️ Error during Groq Final Story ({method_name}): {e}")
        with open(groq_final_story_output_json, "w") as f:
            json.dump({f"error_final_story_for_{method_name}": str(e)}, f, indent=2)
    print(f"Groq Final Story for {method_name} took: {time.time() - start_time_groq2:.2f}s")

print("\n\n================ ALL METHODS PROCESSED ================")