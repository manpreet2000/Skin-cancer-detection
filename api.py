import os
from flask import Flask
from flask import render_template,request

import torch
import albumentations

import numpy as np

import torch.nn as nn

from torch.nn import functional as F

from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

import pretrainedmodels

app=Flask(__name__)
UPLOAD_FOLDER="/home/manpreet/jupyter/SIIM_ISIC_Melanoma_classification/static"
MODEL=None
DEVICE="cpu"

class SEResnext50_32x4d(nn.Module):
    def __init__(self, pretrained='imagenet'):
        super(SEResnext50_32x4d, self).__init__()

        
        self.base_model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=None)
    
        if pretrained is not None:
            self.base_model.load_state_dict(
                torch.load(
                    "/home/manpreet/jupyter/SIIM_ISIC_Melanoma_classification/data/se_resnext50_32x4d-a260b3a4.pth"
                )
            )

        self.l0 = nn.Linear(2048, 1)
    
    def forward(self, image, targets):
        batch_size, _, _, _ = image.shape
        
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        
        out = torch.sigmoid(self.l0(x))
        loss = 0

        return out, loss



def predict(image_path,model):
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    
    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    
    predictions = Engine.predict(test_loader, model, device=DEVICE)
    predictions = np.vstack((predictions)).ravel()

    return predictions

@app.route("/",methods=["GET","POST"])
def upload_predict():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            image_location=os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred=predict(image_location,MODEL)[0]
            print(pred)
            return render_template('index.html',prediction=pred,image_loc=image_file.filename)
    return render_template('index.html',prediction=0,image_loc=None)

if __name__=="__main__":
    MODEL = SEResnext50_32x4d(pretrained='imagenet')
    MODEL.load_state_dict(torch.load("data/model_3.bin",map_location=lambda storage, loc: storage))
    MODEL.to(DEVICE)
    app.run(debug=True)