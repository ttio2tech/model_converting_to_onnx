# model_converting_to_onnx

ONNX format model conversion UI is a webui for converting stable diffusion from ckpt or safetensor to onnx format. 

There is a video demo at https://youtu.be/hE-dSzVSIbI. 

# Installation

Clone the repo to your local PC:

    git clone https://github.com/ttio2tech/model_converting_to_onnx.git
    cd model_converting_to_onnx

In a Python virtual environment, install dependencies: 

    #install pytorch 1.13.1
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
    #install versioned dependencies
    pip install -r requirements.txt

Then start the WebUI:

    python WebUI_convert_model_to_onnx.py


