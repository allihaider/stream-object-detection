import torch
import torchvision
import numpy as np


class Model:
    
    def __init__(self, class_names, detection_threshold, device):
        
        self.class_names = class_names
        self.detection_threshold = detection_threshold
        self.device = device
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        return
        
    def predict(self, images):
        
        inputs = []

        for image in images:
            image = self.transform(image).to(self.device)
            image = image.unsqueeze(0)
            inputs.append(image)

        inputs = torch.cat(inputs)
        self.model.eval().to(self.device)
        outputs = self.model(inputs)

        pred_classes = [self.class_names[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        pred_labels = outputs[0]['labels'].cpu().numpy()
        
        threshold_mask = pred_scores >= self.detection_threshold
        
        classes = np.array(pred_classes)[threshold_mask]
        boxes = pred_bboxes[threshold_mask].astype(np.int32)
        labels = pred_labels[threshold_mask]
        scores = pred_scores[threshold_mask]
        
        return boxes, classes, labels, scores
