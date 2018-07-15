def GetFileInput():
	inputFile = open("secondOrderConditioningInput.txt",'r')
	Iin = []
	for line in inputFile:
		userInput = line.strip()
		# quit = ['quit','q']
		if userInput == EPSILON or userInput == "EPSILON":
			# print ('EPSILON')
			return EPSILON
		elif (userInput.lower() in quit):
			exit()
		elif userInput.lower() == 'status':
			status = m.PrintModel()
			#print(status)
			SaveStatusToFile(status)
			# return GetInput()
			break
		else:
			pairs = userInput.split(',')
			count = 0
			for pair in pairs:
				toks = pair.split(':')
				if len(toks) == 1:
					Iin = Iin +[[toks[0],1.0]]
				else:
					Iin = Iin+[[toks[0],float(toks[1])]]
				count += 1
				return Iin
	# usingFile = False
	# inputFile = None
	# return GetInput()
	return