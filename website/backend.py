import subprocess
import torch
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from lora_diffusion import monkeypatch_lora, tune_lora_scale, patch_pipe
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import numpy as np
from PIL import Image
import glob
import os
#from .models import Img, User, Model
#from . import db
import json
import random
import imageio
import glob
from customqueue import CustomQueue

root_dir = "/home/lamparter/stableDiffusion"

queue = CustomQueue()

pipe = StableDiffusionPipeline.from_pretrained('{0}/test/stable-diffusion-v1-5'.format(root_dir), torch_dtype=torch.float16).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
print("\n\n\n WILLKOMMEN IM BACKEND \n\n\n")
while True:
    queue_entry = queue.dequeue()
    if len(queue_entry) == 0:
        continue
    queue_entry = queue_entry[0]
    if queue_entry[-1] == 1:
        continue
    print("Job received\nID: {0}\nPrompt: {1}\nModel_name: {2}".format(queue_entry[0], queue_entry[1], queue_entry[2]))
    id = queue_entry[0]
    prompt = queue_entry[1]
    model_name = queue_entry[2]
    userid = queue_entry[3]
    steps = queue_entry[4]
    filepath = queue_entry[5]
    imagepath_global = queue_entry[6]
    imagepath_user = queue_entry[7]

    if model_name != 'stable-diffusion-v1-5':
        patch_pipe(pipe, '{0}/StableDiffusionFlask/static/trained_models/{1}/{2}/step_1000.safetensors'.format(root_dir, userid, model_name), patch_text=True, patch_ti=True, patch_unet=True)
        tune_lora_scale(pipe.unet, 0.5)
        # tune_lora_scale(pipe.text_encoder, 0.5)
        print("used fine tuned model: {0}".format(model_name))
        
    if filepath == 'None':
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]   
    else:  
        image = pipe(prompt, num_inference_steps=150, guidance_scale=7, filepath=filepath).images[0]
    
    image.save('{0}/{1}'.format(root_dir, imagepath_global))  
    image.save('{0}/{1}'.format(root_dir, imagepath_user)) 

    queue.finish(id)