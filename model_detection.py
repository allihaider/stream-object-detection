import cv2
import torchvision
import torch
import torchvision.transforms as transforms
import numpy as np
import detect_utils

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

transform = transforms.Compose([transforms.ToTensor(),])
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=int(args['classes']), pretrained=False, min_size=800)
checkpoint = torch.load(f"{args['model']}")
model.load_state_dict(checkpoint['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.eval().to(device)


def predict(image, model, device, detection_threshold):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = transform(image).to(device)
	image = image.unsqueeze(0)
	outputs = model(image)

	pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
	pred_scores = outputs[0]['scores'].detach().cpu().numpy()
	pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
	pred_labels = outputs[0]['labels'].cpu().numpy()
	
	threshold_mask = pred_scores >= detection_threshold
	classes = np.array(pred_classes)[threshold_mask]
	boxes = pred_bboxes[threshold_mask].astype(np.int32)
	labels = pred_labels[threshold_mask]
	scores = pred_scores[threshold_mask]
	
	return boxes, classes, labels, scores


def draw_boxes(boxes, classes, labels, scores, image):
	image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
	
	for i, box in enumerate(boxes):
		color = COLORS[labels[i]]
		cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

	return image


with torch.no_grad():
	boxes, classes, labels, scores = predict(frame, model, device, 0.5)