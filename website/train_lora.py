from flask import Blueprint, render_template, request, flash, redirect, url_for
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from lora_diffusion import monkeypatch_lora, tune_lora_scale, patch_pipe
from flask_login import login_required, current_user
import numpy as np
from PIL import Image
import glob
import os
from lora_diffusion.cli_lora_pti import train as train_lora_source
import uuid
import subprocess

train_lora = Blueprint('train_lora', __name__)
root_dir = "/home/lamparter/stableDiffusion"

def run_train_lora(imagepath, token, template):
    
    train_lora_source(
            instance_data_dir = imagepath,
            pretrained_model_name_or_path = '{0}/test/stable-diffusion-v1-5'.format(root_dir),
            output_dir = '{0}/StableDiffusionFlask/static/trained_models/{1}/{2}'.format(root_dir, current_user.id, token + "_" + str(uuid.uuid4())),
            train_text_encoder = True,
            resolution = 256,
            train_batch_size = 1,
            gradient_accumulation_steps = 4,
            scale_lr = True,
            learning_rate_unet = 1e-4,
            learning_rate_text = 1e-5,
            learning_rate_ti = 5e-4,
            color_jitter = True,
            lr_scheduler = "linear",
            lr_warmup_steps = 0,
            placeholder_tokens = token,
            use_template = template,
            save_steps = 100,
            max_train_steps_ti = 1000,
            max_train_steps_tuning = 1000,
            perform_inversion = True,
            clip_ti_decay = True,
            weight_decay_ti = 0.000,
            weight_decay_lora = 0.001,
            continue_inversion = True,
            continue_inversion_lr = 1e-4,
            device = "cuda:3",
            lora_rank = 1,
            
    )


@train_lora.route("/train_lora", methods=['GET', 'POST'])
@login_required
def train():

    if request.method == 'POST':
        
        save_path = '/home/lamparter/stableDiffusion/StableDiffusionFlask/static/upload_folder/{0}/'.format(current_user.id)
        files=request.files.getlist("files[]")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
            #save training images
            for file in files:
                file.save(os.path.join(save_path, file.filename))  
        else:    
            #remove all files in users upload dir
            for filename in os.listdir(save_path):
                file_path = os.path.join(save_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
            
            #save training images
            for file in files:
                file.save(os.path.join(save_path, file.filename))  
        
        
        run_train_lora(save_path, request.form['token'], request.form['type'])   
        # run_train_dreambooth(save_path, request.form['token'], request.form['type'])    

        return render_template("train_lora.html", user=current_user)
    
    return render_template("train_lora.html", user=current_user)

# @train_lora.route("/train_dreambooth", methods=['GET', 'POST'])
# @login_required
# def train_dreambooth():

#     if request.method == 'POST':
        
#         save_path = '/home/lamparter/stableDiffusion/StableDiffusionFlask/static/upload_folder'
#         files=request.files.getlist("files[]")
        
#         for file in files:
#             file.save(os.path.join(save_path, file.filename))
            
#         text_field_token = request.form['token']
#         token =  str(text_field_token) + "_" + str(uuid.uuid4())    
        
#         run_train_dreambooth(save_path, token, request.form['type'])    

#         return render_template("train_lora.html", user=current_user)
    
#     return render_template("train_lora.html", user=current_user)


# def run_train_dreambooth(imagepath, token, template):
    # MODEL_NAME = '{0}/test/stable-diffusion-v1-5'.format(root_dir)
    # INSTANCE_DIR = imagepath
    # OUTPUT_DIR = '{0}/StableDiffusionFlask/static/trained_models/{1}/{2}'.format(root_dir, current_user.id, token + "_" + str(uuid.uuid4()))
    # # CLASS_DIR = '{0}/StableDiffusionFlask/static/trained_models/dreambooth/class_data'.format(root_dir)

    # script_path = "/home/lamparter/stableDiffusion/StableDiffusionFlask/website/train_dreambooth_lora.py"

    # command = [
    #     script_path,
    #     "--pretrained_model_name_or_path=" + MODEL_NAME,
    #     "--dataset_name=" + INSTANCE_DIR,
    #     "--dataloader_num_workers=8",
    #     "--resolution=512 ",
    #     "--center_crop ",
    #     "--random_flip",
    #     "--train_batch_size=1",
    #     "--gradient_accumulation_steps=4",
    #     "--max_train_steps=15000",
    #     "--learning_rate=1e-04",
    #     "--max_grad_norm=1",
    #     "--lr_scheduler=cosine ",
    #     "--lr_warmup_steps=0",
    #     "--output_dir=" + OUTPUT_DIR,
    #     "--push_to_hub",
    #     "--hub_model_id=" +  token,
    #     "--report_to=wandb",
    #     "--checkpointing_steps=500",
    #     "--validation_prompt=" + token,
    #     "--seed=1337",
    #     "--mixed_precision=fp16"
    # ]

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # subprocess.run(command)
    