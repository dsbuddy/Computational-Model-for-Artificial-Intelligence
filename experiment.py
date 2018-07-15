from engine import *
import sys
import csv

def userPrompt():
	print("Welcome to CFA Experiment Simulation")
	print("Enter \'exit\' to quit.\n")

	userIn = "Enter number for which experiment to conduct:"
	userIn += "\n1) Delayed Conditioning        6) Simultaneous Conditioning"
	userIn += "\n2) Second Order Conditioning   7) Compound Conditioning"
	userIn += "\n3) Latent Inhibition           8) Sensory Preconditioning"
	userIn += "\n4) Extinction                  9) Blocking"
	userIn += "\n5) Partial Reinforcement      10) Extinction in Second Order Conditioning"
	userIn += "\n\nExperiment #"

	try:
		experiment = int(input(userIn))
		fileName = input("\nPlease enter a file name:\n")

		if experiment == 1:
			print("\nStarting Delayed Conditioning . . .")
			delayedConditioning()
		elif experiment == 2:
			print("\nStarting Second Order Conditioning . . .")
			secondOrderConditioning()
		elif experiment == 3:
			print("\nStarting Latent Inhibition . . .")
			latentInhibition()
		elif experiment == 4:
			print("\nStarting Extinction . . .")
			extinction()
		elif experiment == 5:
			print("\nStarting Partial Reinforcement . . .")
			partialReinforcement()
		elif experiment == 6:
			print("\nStarting Simultaneous Conditioning . . .")
			simultaneousConditioning()
		elif experiment == 7:
			print("\nStarting Compound Conditioning . . .")
			compoundConditioning()
		elif experiment == 8:
			print("\nStarting Sensory Preconditioning . . .")
			sensoryPreconditioning()
		elif experiment == 9:
			print("\nStarting Blocking . . .")
			blocking()
		elif experiment == 10:
			print("\nStarting Extinction in Second Order Conditioning . . .")
			extinctionSecondOrder()
		elif experiment == 11:
			print("\nStarting Delayed Conditioning Full . . .")
			delayedConditioningFull(fileName)
		else:
			print("Not a valid experiment")
			sys.exit()
	except ValueError:
		print("Not a valid command")
		sys.exit()

def delayedConditioning():
	main('delayedConditioning.txt', 'delayedConditioningInput.txt', 'delayedOutput.txt')

def delayedConditioningFull(fileName):
	main('delayedConditioning.txt', 'delayedConditioningInput.txt', 'delayedOutput_' + str(fileName) + '.txt')

def secondOrderConditioning():
	main('secondOrderConditioning.txt', 'secondOrderConditioningInput.txt', 'secondOrderOutput.txt')

def latentInhibition():
	main('latentInhibition.txt', 'latentInhibitionInput.txt', 'latentInhibitionOutput.txt')

def extinction():
	pass
	
def partialReinforcement():
	pass
	
def simultaneousConditioning():
	pass
	
def compoundConditioning():
	pass
	
def sensoryPreconditioning():
	pass
	
def blocking():
	pass
	
def extinctionSecondOrder():
	pass


def outputToCSV(extractFile, stateI, stateF, csvFilename):
	# Extract information from output file
	with open(extractFile, 'r') as myfile:
		data = myfile.read().replace('\n', '')
		data = data.strip()

	finalList = []
	strSplit = data.split("------------------------- Trial #")

	for line in strSplit:
		if line.find("State " + str(stateI) + " -> State " + str(stateF)) != -1:
			lastStrLoc = line.find("State " + str(stateI) + " -> State " + str(stateF))
			strToFind = line[:int(lastStrLoc)]
			tempEnd = strToFind[strToFind.rfind("<")+1:]
			outputSymbol = tempEnd[:tempEnd.index(">")]
			finalList.append(str(line[0:line.find(" ")] + ":" + str(outputSymbol)))
	# View output symbols as list
	# print(finalList)

	# Info to CSV
	with open(csvFilename, 'wb') as myfile:
		out = csv.writer(open(csvFilename,"w"), delimiter=',',quoting=csv.QUOTE_ALL)
		out.writerow(finalList)
		print("[Message] Output converted to CSV as \'" + str(csvFilename) + "\'")

if __name__ == "__main__":
	userPrompt()