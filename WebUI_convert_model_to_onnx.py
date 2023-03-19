# Copyright 2023 https://www.youtube.com/@tech-practice9805. All rights reserved.
# txt2img using stable diffusion for AMD GPU on Windows
# The series of tutorials can be found at 
# *****************************************
# https://www.youtube.com/channel/UCC8cMMvUfSYGndUfK7HgCsg
# *****************************************

# python WebUI_AMD_GPU.py -h  # show helps
# Need to provide the path to the model folder(stable_diffusion_onnx). Example usage: 
# python WebUI_AMD_GPU.py -m C:/Users/pentium/stable_diffusion_onnx

# the goal: create the UI to make it easy for model convert:
# ckpt to diffusers, 
# safetensor to diffusers,
# diffusers to onnx
#            
"""
python convert_original_stable_diffusion_to_diffusers_repo.py --from_safetensors --checkpoint_path dreamshaper_332BakedVaeClipFix.safetensors  --dump_path dreamshaper_332BakedVaeClipFix_diffuser --original_config_file v1-inference.yaml   

python convert_stable_diffusion_checkpoint_to_onnx_v1.1.py --model_path="dreamshaper_332BakedVaeClipFix_diffuser" --output_path="dreamshaper_332BakedVaeClipFix_onnx"
"""

# gradio                 3.15.0  -> 3.18.0        pip install -U gradio 
# 1. template for file format convertion
# 2. convert ckpt or safetensor to diffusers 
#errors:
#AttributeError: 'NoneType' object has no attribute 'replace'
#Keyboard interruption in main thread... closing server.
# TODO, download the latest convert script to folder and delete them afterwards?    for the results, show the absolute path for the resulted two folders? 
## TODO, check if the two results folder existed. if do, give a warning. otherwise it will fail to run.  Done 
## TODO, add discord server. Done
## TODO, upgrade diffusers, and the create API for controlnet!   TODO on Ubuntu

import gradio as gr
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
import os 

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# test data: D:\WorkSpace\Downloaded_sd_models\deliberate_v11.ckpt

def conertFn(inputfilepath,input_format,ori_model_version,extract_ema,to_safetensors): 
    from_safetensors = False
    # an option to choose the original format, ckpt or safetensor. 
    
    # if folder exist, return error:
    if os.path.exists('convert_diffuser_model') or os.path.exists('convert_model_onnx_format'):
        return "The results folders exist. please move them or delete them and re-submit. Thanks!"
    
    if '.safetensors' not in inputfilepath and '.ckpt' not in inputfilepath:
        return 'Not a model file! Model file should be a ckpt or safetensor format file. '
    if input_format=='.safetensors':
        from_safetensors = True
    else:
        pass   
    if ori_model_version=='v2':
        image_size = 768
        prediction_type = 'v-prediction'
        upcast_attention=True       
        ori_config_file='v2-inference-v.yaml'       
    else:
        image_size = 512
        prediction_type = 'epsilon'
        upcast_attention=False
        ori_config_file='v1-inference.yaml'  
        
    convert_model_onnx_format = "convert_model_onnx_format"  # folder name for the onnx model

    try:
        pipe = load_pipeline_from_original_stable_diffusion_ckpt(
                checkpoint_path=inputfilepath,
                original_config_file=ori_config_file,
                image_size=image_size,
                prediction_type=prediction_type,
                #model_type=model_type,
                extract_ema=extract_ema,
                #scheduler_type=args.scheduler_type,
                #num_in_channels=args.num_in_channels,
                upcast_attention=upcast_attention,
                from_safetensors=from_safetensors,
            )
        os.mkdir('convert_diffuser_model')   # check if the folder exist or not. 
        pipe.save_pretrained('convert_diffuser_model', safe_serialization=to_safetensors)
        
        # step2: convert diffusers to ONNX format. 
        os.system("python convert_stable_diffusion_checkpoint_to_onnx_v1.1.py --model_path convert_diffuser_model/ --output_path {}".format(convert_model_onnx_format))    
    except:
        return "something went wrong. please try another model file."
    
    return "Model has been successfully converted to ONNX format. Please find the converted diffusers model in {} and the ONNX model in {} . Please rename and move it somewhere else!".format(ROOT_DIR+'\\'+'convert_diffuser_model',ROOT_DIR+'\\'+convert_model_onnx_format)

title = "Turn ckpt or safetensor model into diffusers and onnx format 🚀"
description = """
## Awesome 🖼️ 
<span style="color:blue">   
Why? ONNX format model can <br>
1) enable stable diffusion using <b>AMD GPU</b> on Windows. <br>
2) enable stable diffusion using any CPU. <br>
See <a href='https://youtu.be/hE-dSzVSIbI' target='_blank'> youtube tutorial </a> for details. </span>  <br>
Notes: The converting takes several minutes or more. It requires lots of system RAM. 16GB is recommended, but may not be sufficient. There will likely be lots of warnings in the console which may have an impact for the converted model. YMMV. <br>
For questions or discussions, please <a href="https://discord.gg/SgmBydQ2Mn">click to join the Discord server</a>.
<br> You can also meet with AI bots such as chatGPT bots and stable diffusion bots backed by my 3080Ti GPU! <br>
If you would like to support, here is my <a href="https://ko-fi.com/techpractice"> Kofi link </a> and <a href="https://www.patreon.com/user?u=89548519">Patreon page</a>. Thank you! 
"""
youtube="""
<p style='text-align: left'><a href='https://www.youtube.com/@tech-practice9805' target='_blank'> \
Youtube Channel  Link. Please subscribe to the channel for future videos.</a> </p> 
            <div >
            <img  src='file/discord_server_invite2.png' width="500" height="500" alt='https://discord.gg/SgmBydQ2Mn' >
            </div>
"""
input_format = gr.Radio([".ckpt", ".safetensors"], value=".ckpt", label="Model Format")
ori_model_version = gr.Radio(["v1.x", "v2"], value="v1.x", label="Stable Diffusion version")
extract_ema = gr.Radio(['True','False'], value = 'False', label="Extract EMA")
to_safetensors = gr.Radio(['True','False'], value = 'False', label="Save model in safetensor format")
app  = gr.Interface(fn=conertFn, 
                    inputs=[gr.Textbox(label="Local model path and name, example: D:\WorkSpace\Downloaded_sd_models\elldrethsImagination_v10.ckpt"),
                            input_format,
                            ori_model_version, 
                            extract_ema,
                            to_safetensors], 
                    outputs="text",
                    title=title, 
                    description=description,
                    article=youtube,
                    css="""footer {visibility: hidden}  
                        a {
                        text-decoration: none;
                        color:  #18272F;
                        font-weight: 700;
                        vertical-align: top;}
                        a:hover {
                        color: white;
                        box-shadow: inset 120px 0 0 0 #D65472;
                        }
                        """,
                    allow_flagging="never")

if __name__== "__main__":    
    app.launch(inbrowser=True)   
