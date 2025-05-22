# Video Captioning
Please use the requirements.txt to set up the conda environment/venv.

modified_no_groq.py is the final code base that combines and runs the inference for all the three sampling methods, followed by captioning

image_embedding.py, caption_embedding.py and optic.py has codes for each sampling based captioning method.

finaal_summary.py is used for better formatting and extraction of the data from the summary json files saved, incase the final summary face any error due to formatting


similarity_scores.py used to calculate similarity between ground truth and generated captions


integrate_captioning.py is the file with larger models - use this if there is more gpu memory available. 
