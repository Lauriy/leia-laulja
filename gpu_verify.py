import onnxruntime as ort
import numpy as np
from insightface.app import FaceAnalysis


def verify_gpu_setup():
    """Verify GPU setup after CUDA Toolkit installation."""
    print("=== GPU Verification After CUDA Toolkit Installation ===")
    
    # Check ONNX Runtime providers
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    if 'CUDAExecutionProvider' in providers:
        print("✅ CUDA provider available")
        
        # Test InsightFace GPU initialization
        try:
            print("\n--- Testing InsightFace GPU ---")
            app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Check which provider is actually being used
            print(f"Detection model providers: {app.det_model.providers}")
            
            if 'CUDAExecutionProvider' in str(app.det_model.providers):
                print("🚀 SUCCESS! InsightFace is using GPU!")
                return True
            else:
                print("⚠️  InsightFace still using CPU")
                return False
                
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
    else:
        print("❌ CUDA provider not available")
        return False


if __name__ == "__main__":
    gpu_working = verify_gpu_setup()
    
    if gpu_working:
        print("\n🎉 GPU setup successful! You can now run your main script with GPU acceleration.")
    else:
        print("\n💡 GPU not working yet. Check if you need to:")
        print("   - Restart your IDE/terminal")
        print("   - Reinstall onnxruntime-gpu")
        print("   - Check CUDA installation")