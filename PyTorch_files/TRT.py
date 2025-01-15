import torch
import torch_tensorrt as trt


class TensorRTModelOptimizer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = self.load_model()
        self.optimized_model = None
        
    def load_model(self):
        return torch.jit.load(self.model_dir)
    
    def optimize_model(self):
        # Optimize the model using Torch-TensorRT with FP16 precisiond
        self.optimized_model = trt.compile(
            self.model,
            inputs=[trt.Input(min_shape=(1, 3, 224, 224),
                            opt_shape=(1, 3, 224, 224),
                            max_shape=(16, 3, 224, 224),
                            dtype=torch.half)],
            enabled_precisions={torch.float16}  # Use FP16 precision
        )

    def save_optimized_model(self, save_dir):
        torch.jit.save(self.optimized_model, save_dir)
        print(f"Model başarıyla optimize edildi ve şu konuma kaydedildi: {save_dir}")