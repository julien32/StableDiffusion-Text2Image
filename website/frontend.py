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
from .models import Img, User, Model
from . import db
import json
import random
import imageio
import glob

functions = Blueprint('functions', __name__)

root_dir = "/home/lamparter/stableDiffusion"

@functions.route('/run_lora/<string:prompt>/<string:model_name>/<int:create_gif>')
@login_required
def run_lora(prompt, model_name, create_gif):
    user = current_user
    models = ['stable-diffusion-v1-5'.format(root_dir)]
    models += get_models_of_user(user)
    print(request.data)
    
    print(model_name)
    print(create_gif)
    create_image(prompt, model_name, user, create_gif)
    
    
    # flash('Image created!', category='success')


    return "finished"


def get_models_of_user(user):
    models = glob.glob('{0}/models/{1}/*'.format(root_dir, user))
    models = [model.split('/')[-1] for model in models]
    return models

def get_random_images():
    images = glob.glob('{0}/StableDiffusionFlask/static/generated_images/*'.format(root_dir))
    images.sort(key=os.path.getmtime)
    images.reverse()
    paths = ['/static/generated_images/{}'.format(path.split("/")[-1]) for path in images if path.endswith('.png')]
    # paths = ['/static/generated_images/{0}'.format(path.split("/")[-1]) for path in images]
    random.shuffle(paths)
    
    object_images = []
    for path in paths:
        prompt =  path.split("_")[1].split("/")[1]
        image = Img(image_path=path, prompt=prompt, user_id=current_user.id)
        object_images.append(image)
    
    return object_images

def create_image(prompt, model_name, user, create_gif):
    path_for_gifs = "static/generated_images/{0}/gif".format(current_user.id)
    if not os.path.exists(path_for_gifs):
        os.makedirs(path_for_gifs)
    imagename = '{0}_{1}_{2}.png'.format(prompt, user.id, random.randint(1, 10000000000))
    image_save_path_global = "static/generated_images/" + imagename
    image_save_path_user = "static/generated_images/{0}/".format(current_user.id) + imagename
    filepath = 'None'
    if create_gif == 1:
        filepath = path_for_gifs + '/'
    customqueue = CustomQueue()
    customqueue.enqueue(
        prompt, 
        model_name,
        current_user.id,
        image_save_path_global,
        image_save_path_user,
        num_inference_steps=150, 
        filepath=filepath, 
        wait=True) 
    add_note_to_db(imagename, prompt, model_name)
    if create_gif == 1:
        create_gif_image(path_for_gifs)
        
def create_gif_image(path):
    full_path = '{0}/StableDiffusionFlask/{1}/'.format(root_dir, path)
    images = []
    # filenames = [os.path.basename(file) for file in os.listdir(full_path) if file.endswith('.jpg')]
    directory = '/home/lamparter/stableDiffusion/StableDiffusionFlask/static/generated_images/{0}/gif/'.format(current_user.id)

    filenames = [os.path.basename(file) for file in os.listdir(directory) if file.endswith('.jpg')]
    print(filenames)
    for filename in filenames:
        images.append(imageio.imread("static/generated_images/{0}/gif/{1}".format(current_user.id, filename)))
    imageio.mimsave('{0}/StableDiffusionFlask/{1}/gif.gif'.format(root_dir, path), images)

def add_note_to_db(path, prompt, model_name):
    new_image = Img(image_path=path, prompt=prompt, model=model_name, user_id=current_user.id)
    db.session.add(new_image)
    db.session.commit()
    
def add_model_to_db(name):
    new_model = Model(name=name)
    db.session.add(new_model)
    db.session.commit()
    
    
def get_users_trained_models():
    path = './static/trained_models/{1}'.format(root_dir, current_user.id)
    
    filenames = []
    
    if os.path.exists(path):
        for filename in os.listdir(path):
            filenames.append(filename)
            # underscore_index = filename.find('_')
            # if underscore_index != -1:
            #     filenames.append(filename[:underscore_index])
            # else:
            #     filenames.append(filename)
        
        
    return filenames