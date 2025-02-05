from fastapi import FastAPI
import torch


app = FastAPI()


@app.get("/")
async def root():
    if torch.cuda.is_available():
        version = torch.version.cuda
        gpu_name = torch.cuda.get_device_name(0)
        no_of_gpus = torch.cuda.device_count()
        return {"message": f"CUDA Available!! {version}, {gpu_name}, {no_of_gpus}"}
    
    return {"message": "CUDA not available"}