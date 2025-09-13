from matplotlib import image
import torch
import torchaudio.transforms as T
import torchaudio.functional as AF
import torchvision.transforms.functional as TVF
import torch.nn.functional as TF
from scipy.signal import butter, lfilter


class ECGToSpectogram:
    def __init__(self, sampling_rate, n_fft, hop_length, win_length=None, output_size = None, apply_filter = False, filter_order = 4, cutoff_freq = 40.0):
        self.sampling_rate = sampling_rate
        self.apply_filter = apply_filter
        self.output_size = output_size
        
        self.filter_order = filter_order
        self.cutoff_freq = cutoff_freq
        
        if self.apply_filter:
            self.b, self.a = butter(
                N=self.filter_order,
                Wn=self.cutoff_freq,
                btype='low',
                analog=False,
                fs=self.sampling_rate
            )
            
        self.spectogram_transform = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length if win_length is not None else n_fft,
            hop_length=hop_length,
            power=2,
            window_fn=torch.hann_window
        )
        
        self.db_transform = T.AmplitudeToDB()
    
    def __call__(self, ecg_tensor):
        if self.apply_filter:
            ecg_numpy = ecg_tensor.numpy()
            filtered_ecg_numpy = lfilter(self.b, self.a, ecg_numpy, axis=-1)
            processed_tensor = torch.from_numpy(filtered_ecg_numpy.copy()).float()
        else:
            processed_tensor = ecg_tensor
        
        spectogram = self.spectogram_transform(processed_tensor)
        
        db_spectogram = self.db_transform(spectogram)
        
        if self.output_size:
            C, F, T_steps = db_spectogram.shape
            spectogram = db_spectogram.view(2, 4, F, T_steps)
            spectogram = spectogram.permute(0, 2, 1, 3)
            tiled_spectogram = spectogram.reshape(1, 2 * F, 4 * T_steps)
           
            _, original_height, original_width = tiled_spectogram.shape
            target_height = self.output_size
            target_width = int(original_width * (target_height / original_height)) 
            resized_spectogram = TVF.resize(tiled_spectogram, size=[target_height, target_width], antialias=True)
            
            current_width = resized_spectogram.shape[2]
            padding_needed = self.output_size - current_width
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            
            padded_spectogram = TF.pad(resized_spectogram, (pad_left, pad_right, 0, 0), "constant", 0)
            
            final_image = padded_spectogram.repeat(3, 1, 1)
        else:
            final_image = db_spectogram 
        

        return final_image

class InverseECGSpectogram:
    def __init__(self, n_fft, hop_length, original_tiled_shape, win_length=None):
        self.original_tiled_shape = original_tiled_shape

        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length if win_length is not None else n_fft,
            hop_length=hop_length,
            power=1.0,
            window_fn=torch.hann_window
        )
    
    def __call__(self, image_tensor):
        spectrogram = image_tensor[0:1, :, :]
        
        resized_width = int(self.original_tiled_shape[1] * (image_tensor.shape[1] / self.original_tiled_shape[0]))
        total_padding = image_tensor.shape[2] - resized_width
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        
        if pad_right > 0:
            cropped_spectrogram = spectrogram[:, :, pad_left:-pad_right]
        else:
            cropped_spectrogram = spectrogram[:, :, pad_left:]
            
        rescaled_spectrogram = TVF.resize(cropped_spectrogram, size=list(self.original_tiled_shape), antialias=True)
        
        _, H, W = rescaled_spectrogram.shape
        F_dim = H // 2
        T_dim = W // 4
        
        db_spectrograms = rescaled_spectrogram.view(1, 2, F_dim, 4, T_dim).permute(0, 1, 3, 2, 4).reshape(8, F_dim, T_dim)
        
        power_spectograms = AF.DB_to_amplitude(db_spectrograms, ref=1.0, power=1.0) 
        mag_spec = torch.sqrt(torch.clamp(power_spectograms, min=0.0))
        
        reconstructed_leads = self.griffin_lim(mag_spec)
        
        return reconstructed_leads