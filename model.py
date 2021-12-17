import cv2
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np


class Model:
    
    def __init__(self, class_names, detection_threshold, device):
        
        self.class_names = class_names
        self.detection_threshold = detection_threshold
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor())
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        return
        
    def predict(self, images):
        
        inputs = []

        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).to(device)
            image = image.unsqueeze(0)
            inputs.append(image)

        inputs = torch.cat(inputs)
        outputs = model(inputs)

        pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()
        
        threshold_mask = pred_scores >= self.detection_threshold
        
        classes = np.array(pred_classes)[threshold_mask]
        boxes = pred_bboxes[threshold_mask].astype(np.int32)
        labels = pred_labels[threshold_mask]
        scores = pred_scores[threshold_mask]
        
        return boxes, classes, labels, scores
