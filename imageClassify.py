import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F

#class ToyDataset(Dataset):
#	def __init__(self, x, y):
#		self.features = x
#		self.labels = y
#
#	def __getitem__(self, index):	#instruction for retrieving 1 data record with corresponding label
#		oneX = self.features[index]
#		oneY = self.labels[index]
#		return oneX, oneY
#
#	def __len__(self):
#		return self.labels.shape[0];	#returns len of dataset
#



class NeuralNetwork(torch.nn.Module):
	def __init__(self, numInputs, numOutputs):
		super().__init__()

		self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples by 2
		# 1st hidden layer
		self.conv1 = torch.nn.Conv2d(in_channels=numInputs, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.relu1 = torch.nn.ReLU()

		# 2nd hidden layer
		self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
		self.relu2 = torch.nn.ReLU()

		#output layer
		self.fc = torch.nn.Linear(64 * 16 * 16, numOutputs)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pool(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool(x)

		x = x.view(x.size(0), -1)

		logits = self.fc(x)
		return logits



def computeAccuracy(model, dataLoader):
	model = model.eval()
	correct = 0.0
	totalExamples = 0

	for idx, (features, labels) in enumerate(dataLoader):
		with torch.no_grad():
			logits = model(features)

		predictions = torch.argmax(logits, dim = 1)
		compare = labels == predictions
		correct += torch.sum(compare)	#sum the number of trues
		totalExamples += len(compare)	#count total of examples

	return (correct / totalExamples).item()	#.item returns the percent as a float


def train(numEpochs, optimizer, trainLoader):

	for epoch in range(numEpochs):


		model.train()


		for batchIdx, (features, labels) in enumerate(trainLoader):
			features, labels = features.to(device), labels.to(device)	#transfers the data to the GPU
			logits = model(features)

			loss = F.cross_entropy(logits, labels)

			optimizer.zero_grad()	#sets the gradient to 0 to prevent gradient accumulation
			loss.backward()	#computes the gradient of the loss given the models parameters
			optimizer.step()	#the optimizer uses the gradient to update the models parameters

			###LOGGING
			print(f"Epoch: {epoch+1:03d}/{numEpochs:03d}"
			      f" | Batch {batchIdx:03d}/{len(trainLoader) : 03d}"
			      f" | Train Loss: {loss:.2f}")

			model.eval()
			#insert optional model evaluation code



		torch.save(model.state_dict(), f"imageClassifygen{epoch}.pth")



if __name__ == "__main__":

	transform = transforms.Compose([
		transforms.Resize((64, 64)),	#resizes image to 64x64
		transforms.ToTensor(),	#convert image to tensor
		transforms.Normalize((0.5,), (0.5,))	#Normalize to [-1, 1]
	])

	#load datasets
	trainDataset = ImageFolder(root = "dataset/train", transform=transform)
	testDataset = ImageFolder(root = "dataset/test", transform=transform)

	#dataloader
	trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True)
	testLoader = DataLoader(testDataset, batch_size=16, shuffle=False)

	#check class labels
	print(trainDataset.class_to_idx)

	model = NeuralNetwork(numInputs = 3, numOutputs = 10)	#the dataset has 2 features and 2 classes

	#model.load_state_dict(torch.load("imageClassifygen4.pth"))


	device = torch.device("cpu")	#defines a device that defaults to the GPU
	model = model.to(device)	#transfers the model to the GPU


	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)	#the optimizer needs to know which parameters to optimize

	train(10, optimizer, trainLoader)

	print(computeAccuracy(model, testLoader))

	torch.save(model.state_dict(), "imageClassifyFinal.pth")
