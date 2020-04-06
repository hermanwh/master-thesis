def getConfig(dirr, model, res):
	dirrs = {
		'D': {
			'A': getConfigD_modelA,
			'B': getConfigD_modelB,
			'C': getConfigD_modelC,
			None: getConfigD,
		},
		'F': {
			'A': getConfigF_modelA,
			'B': getConfigF_modelB,
			None: getConfigD,
		},
		'G': {
			'A': getConfigG_modelA,
			'B': getConfigG_modelB,
			'C': getConfigG_modelC,
			None: getConfigD,
		},
	}

	return dirrs[dirr][model](res)

def getConfigD(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/D/dataC.csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigD_columns()

	# List of column names to ignore completely
	irrelevantColumns = [

	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2020-01-01 00:00:00", "2020-03-20 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2020-01-01 00:00:00",
		"2020-08-01 00:00:00"
	]

	columnOrder = ['20TT001', '20TT002', '20FT001', '50TT001', '50TT002', '50TV001', '50FT001', '20PDT001', '50PDT001', '20PT001', '50PT001']

	return [filename, columns, irrelevantColumns, None, traintime, testtime, columnOrder]

def getConfigD_modelA(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/D/dataC.csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigD_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
		'20PT001',
		'50PT001',
		'50FT001',
		'50PDT001',
		'50TV001',
		'20PDT001',
	]

	# List of column names used a targets
	targetColumns = [
		'50TT002',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2020-01-01 00:00:00", "2020-03-20 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2020-01-01 00:00:00",
		"2020-08-01 00:00:00"
	]

	columnOrder = ['20FT001', '20TT001', '20TT002', '50TT001', '50TT002']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigD_modelB(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/D/dataC.csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigD_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
		'20PT001',
		'50PT001',
		'50FT001',
		'50PDT001',
		'20PDT001',
	]

	# List of column names used a targets
	targetColumns = [
		'50TT002',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2020-01-01 00:00:00", "2020-03-20 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2020-01-01 00:00:00",
		"2020-08-01 00:00:00"
	]

	columnOrder = ['20FT001', '20TT001', '20TT002', '50TT001', '50TV001', '50TT002']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigD_modelC(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/D/dataC.csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigD_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
		'20PT001',
		'50PT001',
		'50FT001',
		'50PDT001',
		'50TV001',
	]

	# List of column names used a targets
	targetColumns = [
		'50TT002',
		'20PDT001',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2020-01-01 00:00:00", "2020-03-20 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2020-01-01 00:00:00",
		"2020-08-01 00:00:00"
	]

	columnOrder = ['20FT001', '20TT001', '20TT002', '50TT001', '50TT002', '20PDT001']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigF(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/F/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigF_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
			'FT0111',
			'PDT0108_MA_Y',
			'PDT0119_MA_Y',
			'PDT0118_MA_Y',
			'TT0104_MA_Y',
			'TIC0103_CA_YX',
			'TI0115_MA_Y',
			'TT0652_MA_Y',
			'TIC0103_CA_Y',
			'PIC0104_CA_YX',
			'TIC0101_CA_Y',
			'TT0102_MA_Y',
			'TIC0101_CA_YX',
			'TT0651_MA_Y',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2018-01-01 00:00:00",
		"2019-05-01 00:00:00"
	]
	
	columnOrder = ['TT0106_MA_Y', 'TIC0105_CA_YX', 'FYN0111', 'TIC0425_CA_YX', 'TT0653_MA_Y', 'TIC0105_CA_Y']

	return [filename, columns, irrelevantColumns, None, traintime, testtime, columnOrder]

def getConfigF_modelA(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/F/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigF_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
			'FT0111',
			'PDT0108_MA_Y',
			'PDT0119_MA_Y',
			'PDT0118_MA_Y',
			'TT0104_MA_Y',
			'TIC0103_CA_YX',
			'TI0115_MA_Y',
			'TT0652_MA_Y',
			'TIC0103_CA_Y',
			'PIC0104_CA_YX',
			'TIC0101_CA_Y',
			'TT0102_MA_Y',
			'TIC0101_CA_YX',
			'TT0651_MA_Y',
			'TIC0105_CA_Y',
	]

	# List of column names used a targets
	targetColumns = [
		'TT0653_MA_Y',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2018-01-01 00:00:00",
		"2019-05-01 00:00:00"
	]

	columnOrder = ['FYN0111', 'TT0106_MA_Y', 'TIC0105_CA_YX', 'TIC0425_CA_YX', 'TT0653_MA_Y']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigF_modelB(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/F/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigF_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
			'FT0111',
			'PDT0108_MA_Y',
			'PDT0119_MA_Y',
			'PDT0118_MA_Y',
			'TT0104_MA_Y',
			'TIC0103_CA_YX',
			'TI0115_MA_Y',
			'TT0652_MA_Y',
			'TIC0103_CA_Y',
			'PIC0104_CA_YX',
			'TIC0101_CA_Y',
			'TT0102_MA_Y',
			'TIC0101_CA_YX',
			'TT0651_MA_Y',
	]

	# List of column names used a targets
	targetColumns = [
		'TT0653_MA_Y',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2018-01-01 00:00:00",
		"2019-05-01 00:00:00"
	]

	columnOrder = ['FYN0111', 'TT0106_MA_Y', 'TIC0105_CA_YX', 'TIC0425_CA_YX', 'TIC0105_CA_Y', 'TT0653_MA_Y']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigG(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/G/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = getConfigG_columns()

	# List of column names to ignore completely
	irrelevantColumns = [

	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2017-01-01 00:00:00",
		"2020-03-01 00:00:00",
	]
	
	columnOrder = ['TZI0012', 'TI0066', 'FI0010', 'TT0025', 'TT0026', 'TIC0022U', 'FI0027', 'PDI0064', 'PDT0024', 'PI0001']

	return [filename, columns, irrelevantColumns, None, traintime, testtime, columnOrder]

def getConfigG_modelA(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/G/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = columns = getConfigG_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
		'PI0001',
		'FI0027',
		'TIC0022U',
		'PDT0024',
		'PDI0064',
	]

	# List of column names used a targets
	targetColumns = [
		'TT0026',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2017-01-01 00:00:00",
		"2020-03-01 00:00:00",
	]

	columnOrder = ['FI0010', 'TZI0012', 'TI0066', 'TT0025', 'TT0026']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigG_modelB(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/G/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = columns = getConfigG_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
		'PI0001',
		'FI0027',
		'PDI0064',
		'PDT0024',
	]

	# List of column names used a targets
	targetColumns = [
		'TT0026',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2017-01-01 00:00:00",
		"2020-03-01 00:00:00",
	]

	columnOrder = ['FI0010', 'TZI0012', 'TI0066', 'TT0025', 'TIC0022U','TT0026']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigG_modelC(res):
	# File path to dataset .csv file
	filename = "../master-thesis-db/datasets/G/data_" + res + ".csv"

	# List of columns on form ['name', 'desc', 'unit']
	columns = columns = getConfigG_columns()

	# List of column names to ignore completely
	irrelevantColumns = [
		'PI0001',
		'FI0027',
		'TIC0022U',
		'PDT0024',
	]

	# List of column names used a targets
	targetColumns = [
		'TT0026',
		'PDI0064',
	]

	# List of training periods on form ['start', 'end']
	traintime = [
		["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
	]

	# Testing period, recommended: entire dataset
	testtime = [
		"2017-01-01 00:00:00",
		"2020-03-01 00:00:00",
	]

	columnOrder = ['FI0010', 'TZI0012', 'TI0066', 'TT0025', 'TT0026', 'PDI0064']

	return [filename, columns, irrelevantColumns, targetColumns, traintime, testtime, columnOrder]

def getConfigD_columns():
	return [
		['20TT001', 'Gas side inlet temperature', 'degrees'],
		['20PT001', 'Gas side inlet pressure', 'barG'],
		['20FT001', 'Gas side flow', 'M^3/s'],
		['20TT002', 'Gas side outlet temperature', 'degrees'],
		['20PDT001', 'Gas side pressure difference', 'bar'],
		['50TT001', 'Cooling side inlet temperature', 'degrees'],
		['50PT001', 'Cooling side inlet pressure', 'barG'],
		['50FT001', 'Cooling side flow', 'M^3/s'],
		['50TT002', 'Cooling side outlet temperature', 'degrees'],
		['50PDT001', 'Cooling side pressure differential', 'bar'],
		['50TV001', 'Cooling side valve opening', '%'],
	]

def getConfigF_columns():
	return [
		['FYN0111', 'Gasseksport rate', 'MSm^3/d'],
		['FT0111', 'Gasseksport molvekt','g/mole'],
		['TT0102_MA_Y', 'Varm side A temperatur inn', 'degrees'],
		['TIC0101_CA_YX', 'Varm side A temperatur ut', 'degrees'],
		['TT0104_MA_Y', 'Varm side B temperatur inn', 'degrees'],
		['TIC0103_CA_YX', 'Varm side B temperatur ut', 'degrees'],
		['TT0106_MA_Y', 'Varm side C temperatur inn', 'degrees'],
		['TIC0105_CA_YX', 'Varm side C temperatur ut', 'degrees'],
		['TI0115_MA_Y', 'Scrubber temperatur ut', 'degrees'],
		['PDT0108_MA_Y', 'Varm side A trykkfall', 'Bar'],
		['PDT0119_MA_Y', 'Varm side B trykkfall', 'Bar'],
		['PDT0118_MA_Y', 'Varm side C trykkfall', 'Bar'],
		['PIC0104_CA_YX', 'Innløpsseparator trykk', 'Barg'],
		['TIC0425_CA_YX', 'Kald side temperatur inn', 'degrees'],
		['TT0651_MA_Y', 'Kald side A temperatur ut', 'degrees'],
		['TT0652_MA_Y', 'Kald side B temperatur ut', 'degrees'],
		['TT0653_MA_Y', 'Kald side C temperatur ut', 'degrees'],
		['TIC0101_CA_Y', 'Kald side A ventilåpning', '%'],
		['TIC0103_CA_Y', 'Kald side B ventilåpning', '%'],
		['TIC0105_CA_Y', 'Kald side C ventilåpning', '%'],
	]

def getConfigG_columns():
	return [
		['PDI0064', 'Process side dP', 'bar'],
		['TI0066', 'Process side Temperature out','degrees'],
		['TZI0012', 'Process side Temperature in', 'degrees'],
		['FI0010', 'Process side flow rate', 'MSm^3/d(?)'],
		['TT0025', 'Cooling side Temperature in', 'degrees'],
		['TT0026', 'Cooling side Temperature out', 'degrees'],
		['PI0001', 'Cooling side Pressure in', 'barG'],
		['FI0027', 'Cooling side flow rate', 'MSm^3/d(?)'],
		['TIC0022U', 'Cooling side valve opening', '%'],
		['PDT0024', 'Cooling side dP', 'bar'],
	]