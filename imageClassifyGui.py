import pygame
import sys
import imageClassify
import torch

def getInput():
	pygame.init()
	black = (0, 0, 0)
	white = (255, 255, 255)
	blue = (0, 0, 255)
	width, height = 500, 600
	screen = pygame.display.set_mode((width, height))
	drawing = False

	font = pygame.font.SysFont('NotoSans-Regular', 35)

	text = font.render('Guess', True, black)
	screen.fill(black)


	pygame.draw.rect(screen, blue, [0, 500, width, 100])

	running = True
	while running:
		mouseLoc = pygame.mouse.get_pos()	#stores the mouse location as a tuple (x, y)
		screen.blit(text, (200, 550))	#put text on screen
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if(mouseLoc[1] >= 500):
					running = False
				drawing = True
			elif event.type == pygame.MOUSEBUTTONUP:
				drawing = False
			elif event.type == pygame.MOUSEMOTION and drawing and mouseLoc[1] < 500:
				pygame.draw.circle(screen, white, event.pos, 30)
	    
		#pygame.display.flip()
		pygame.display.update()

	pygame.image.save(screen, "test/drawing.png")  # Save drawing before closing
	pygame.quit()

def cropImage():
	import cv2

	image = cv2.imread("test/drawing.png")

	x1, x2 = 0, 500	#defining croping region
	y1, y2 = 0, 500

	croppedImage = image[y1: y2, x1:x2]

	cv2.imwrite("test/drawing.png", croppedImage)

def guess():
	model = imageClassify.NeuralNetwork(numInputs = 3, numOutputs = 10)    #the dataset has 3 features and 420 classes
	 
	model.load_state_dict(torch.load("imageClassifyFinal.pth"))

	import torchvision.transforms as transforms

	transform = transforms.Compose([
		transforms.Resize((64, 64)),	#resizes image to 64x64
		transforms.ToTensor(),	#convert image to tensor
		transforms.Normalize((0.5,), (0.5,))	#Normalize to [-1, 1]
	])

	from PIL import Image

	image = Image.open("test/drawing.png")

	transformedImage = transform(image)

	# Add a batch dimension (models expect [batch_size, channels, height, width])
	transformedImage = transformedImage.unsqueeze(0)  # Shape: [1, C, H, W]

	with torch.no_grad():
		logits = model(transformedImage)

	prediction = torch.argmax(logits, dim = 1)

	print(int(prediction))


if __name__ == "__main__":
	getInput()
	cropImage()
	guess()
