import json
import random
import numpy as np
import torch
from diffusers import StableDiffusionPipeline 
from ecg_spect_diff.dataset.dataset import getKCLTrainTestDataset
from ecg_spect_diff.utils import (
    resize_and_stack_images,
    load_generator_model,
    plot_spectrograms_comparison,
    plot_overlapping_ecgs
)
from ecg_spect_diff.dataset.dataset import SpectrogramExtractor
import matplotlib.pyplot as plt


datasetConfig = {
        "data_path": "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/ecg_spect_diff/parquet_patients/ecgs_patients.parquet",
        "dataDir": "/uufs/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/",
        "timeCutOff": 1800,
        "lowerCutOff": 0,
        "randSeed": 7777,
        "scale_training_size": 1.0,
        "kcl_params": {
            "lowThresh": 4.0,
            "highThresh": 5.0,
            "highThreshRestrict": 8.5
    }
}

class Inference:
    def __init__(self, checkpoint_path, vocoder_path):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.pipeline = StableDiffusionPipeline.from_pretrained(checkpoint_path).to(self.device)

        with open(f'{checkpoint_path}/normalization_params.json', 'r') as f:
            params = json.load(f)
            self.global_min = params['global_min']
            self.global_max = params['global_max']
        
        self.generator = load_generator_model(vocoder_path)
        self.generator = self.generator.to(self.device)
        self.generator.eval()
        self.spectrogram_extractor = SpectrogramExtractor()
    
    def normalize_spectrogram(self, spec): 
        spec_0_1 = (spec - self.global_min) / (self.global_max - self.global_min)
        spec_neg1_1 = spec_0_1 * 2.0 - 1.0
        return spec_neg1_1

    def denormalize_spectrogram(self, spec):
        spec_0_1 = (spec + 1.0) / 2.0
        original_spec = spec_0_1 * (self.global_max - self.global_min) + self.global_min
        return original_spec

    def run_img2img(self, spect, mod_prompt):
        spect = self.normalize_spectrogram(spect)
        
        generator = torch.Generator(device=self.device).manual_seed(42)
        result = self.pipeline(
            prompt=mod_prompt,
            height=256,
            width=256,
            image=spect,
            strength=0.1,
            num_inference_steps=10,
            guidance_scale=3.0,
            negative_prompt=None,
            num_images_per_prompt=1,
            generator=None,
        )

        images = result.images
        images = resize_and_stack_images(result.images)
        images = images * 2.0 - 1.0
        images = self.denormalize_spectrogram(images)

        return images
    
    def _get_ecgs_from_spectrograms(self, images):
        B = images.shape[0]
        IMG_SIZE = 256
        images = images.view(-1, IMG_SIZE, IMG_SIZE).to(self.device) # B*3, 256, 256
        out = self.spectrogram_extractor(images) # B * 3, 8, 65, 126
        out = out.view(-1, 65, 126) # B * 3 * 8, 65, 126
        with torch.no_grad():
            res = self.generator(out) # B * 3 * 8, 1, 2500
        final_ecgs = res.view(B, -1, 8, 2500).detach().cpu() # B , 3, 8, 2500
        return final_ecgs
    
    def generate_ecgs_from_prompts(self, prompts, num_ecgs, output_path="generated_ecgs.pth", num_inference_steps=100, guidance_scale=3.0, batch_size=8):
        """
        Generate ECGs from prompts using batched inference for efficiency.
        
        Args:
            prompts: List of prompt strings to choose from
            num_ecgs: Total number of ECGs to generate
            output_path: Path to save the generated ECGs
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            batch_size: Number of ECGs to generate per batch (default: 8)
        
        Returns:
            Dictionary with 'ecgs' tensor (num_ecgs, 3, 8, 2500) and 'prompts' list
        """
        if not prompts:
            raise ValueError("Prompts list cannot be empty")
        
        IMG_SIZE = 256
        all_ecgs = []
        used_prompts = []
        
        print(f"Generating {num_ecgs} ECGs from {len(prompts)} prompts using batch size {batch_size}...")
        
        # Process in batches
        num_batches = (num_ecgs + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Determine batch size for this iteration
            current_batch_size = min(batch_size, num_ecgs - batch_idx * batch_size)
            batch_start = batch_idx * batch_size
            
            # Select prompts for this batch
            batch_prompts = [random.choice(prompts) for _ in range(current_batch_size)]
            used_prompts.extend(batch_prompts)
            
            print(f"Processing batch {batch_idx + 1}/{num_batches} (ECGs {batch_start + 1}-{batch_start + current_batch_size}/{num_ecgs})...")
            
            # Generate spectrograms in batch using pipeline
            with torch.no_grad():
                result = self.pipeline(
                    prompt=batch_prompts,
                    height=IMG_SIZE,
                    width=IMG_SIZE,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=None,
                    num_images_per_prompt=1,
                )
            
            # Convert PIL images to tensor batch: (current_batch_size, 3, 256, 256)
            images = resize_and_stack_images(result.images)
            images = images * 2.0 - 1.0  # Convert from [0,1] to [-1,1]
            images = self.denormalize_spectrogram(images)  # Denormalize to original range
            
            # Convert spectrograms to ECGs in batch: (current_batch_size, 3, 8, 2500)
            batch_ecgs = self._get_ecgs_from_spectrograms(images)
            
            # Store ECGs from this batch (already on CPU from _get_ecgs_from_spectrograms)
            all_ecgs.append(batch_ecgs)
            
            # Clear GPU cache periodically to manage memory
            if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Stack all ECGs: (num_ecgs, 3, 8, 2500)
        ecgs_tensor = torch.cat(all_ecgs, dim=0)
        
        save_data = {
            'ecgs': ecgs_tensor,  # Shape: (num_ecgs, 3, 8, 2500)
            'prompts': used_prompts  # List of prompts used for each ECG
        }
        
        torch.save(save_data, output_path)
        print(f"Saved {num_ecgs} ECGs to {output_path}")
        print(f"ECG shape: {ecgs_tensor.shape}")
        print(f"Prompts saved: {len(used_prompts)}")
        
        return save_data
    
    def infer(self, og_spect, og_prompt, mod_prompts, og_ecg=None):
        # extract modified spectrograms
        images = self.run_img2img(og_spect, mod_prompts)
        spect_fig = plot_spectrograms_comparison(og_spect, images, og_prompt, mod_prompts)

        # extract ecgs
        all_spects = torch.cat([og_spect.unsqueeze(0), images]) # has one original spect, rest modified
        extracted_ecgs = self._get_ecgs_from_spectrograms(all_spects)
 
        # Collect ECGs
        all_ecg_figs = []
        if og_ecg is not None:
            og_ecg = og_ecg.unsqueeze(0) # 1, 8, 2500
        og_ecg_spect = extracted_ecgs[0][0].unsqueeze(0) # 1, 8, 2500
        for i in range(extracted_ecgs.shape[0]-1):
            curr_ecg = extracted_ecgs[i+1] # 3, 8, 2500
            curr_ecgs = torch.cat([og_ecg_spect, curr_ecg])
            if og_ecg is not None:
                curr_ecgs = torch.cat([og_ecg, curr_ecgs])
            ecg_fig = plot_overlapping_ecgs(curr_ecgs, label=f"ECG from Spectrogram {i+1}")
            all_ecg_figs.append(ecg_fig)
            
        # plot stuff
        spect_fig.savefig(f"spectrograms_comparison.png")
        for i, fig in enumerate(all_ecg_figs):
            fig.savefig(f'ecg_plot_{i+1}.png')

def main():
    path = "/uufs/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg-spect/ecg_spect_diff/checkpoints/2025-10-19 17:31:37.772008/pipeline_checkpoint-19-0"
    vocoder_path = "spect_ecg_gan/checkpoints/2025-09-14_01-39-31/00200000/g_00200000"
    
    inference = Inference(path, vocoder_path)

    # train_dataset, test_dataset = getKCLTrainTestDataset(datasetConfig)
    # data = train_dataset[0]
    # for i, inst in enumerate(train_dataset):
    #     print(f'Looking at data instanec {i}', end="\r") 
    #     if inst['val'] >= 4.5:
    #         data = inst
    #         break

    # spect = data['image']
    # mod_prompt = "ECG that shows Normal Signs of Hyperkalemia"
    mod_prompts = ["ECG that shows Normal Signs of Hyperkalemia", 
                   "ECG that shows Moderate Signs of Hyperkalemia", 
                   "ECG that shows Severe Signs of Hyperkalemia",
                   "ECG that shows Extremely Severe Signs of Hyperkalemia"] 
    # og_prompt = data['cond_inputs']['text']
    # og_ecgs = data['ecgs']
    # print(og_prompt)
    # inference.infer(spect, og_prompt, mod_prompts=mod_prompts, og_ecg=og_ecgs)

    inference.generate_ecgs_from_prompts(prompts=mod_prompts, num_ecgs=50000, 
                                         output_path="generated_ecgs/50k_gen2.pth",
                                         batch_size=256 * 2)


if __name__ == "__main__":
    main()