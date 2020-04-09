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
			None: getConfigF,
		},
		'G': {
			'A': getConfigG_modelA,
			'B': getConfigG_modelB,
			'C': getConfigG_modelC,
			None: getConfigG,
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
		['20TT001', 'Process Inlet Temperature', 'Degrees'],
		['20PT001', 'Process Inlet Pressure', 'Bar'],
		['20FT001', 'Process Flow Rate', 'kg/hour'],
		['20TT002', 'Process Outlet Temperature', 'Degrees'],
		['20PDT001', 'Process Pressure Difference', 'Bar'],
		['50TT001', 'Coolant Inlet Temperature', 'Degrees'],
		['50PT001', 'Coolant Inlet Pressure', 'Bar'],
		['50FT001', 'Coolant Flow Rate', 'kg/hour'],
		['50TT002', 'Coolant Outlet Temperature', 'Degrees'],
		['50PDT001', 'Coolant Pressure Difference', 'Bar'],
		['50TV001', 'Coolant Valve Opening', '%'],
	]

def getConfigF_columns():
	return [
		['FYN0111', 'Process Flow Rate', 'MSm^3/day'],
		['FT0111', 'Process Flow Molecular Weight','g/mole'],
		['TT0102_MA_Y', 'Process Inlet Temperature A', 'Degrees'],
		['TIC0101_CA_YX', 'Process Outlet Temperature A', 'Degrees'],
		['TT0104_MA_Y', 'Process Inlet Temperature B', 'Degrees'],
		['TIC0103_CA_YX', 'Process Outlet Temperature B', 'Degrees'],
		['TT0106_MA_Y', 'Process Inlet Temperature C', 'Degrees'],
		['TIC0105_CA_YX', 'Process Outlet Temperature C', 'Degrees'],
		['TI0115_MA_Y', 'Scrubber Outlet Temperature', 'Degrees'],
		['PDT0108_MA_Y', 'Process A Pressure Difference', 'Bar'],
		['PDT0119_MA_Y', 'Process B Pressure Difference', 'Bar'],
		['PDT0118_MA_Y', 'Process C Pressure Difference', 'Bar'],
		['PIC0104_CA_YX', 'Separator Inlet Pressure', 'Bar'],
		['TIC0425_CA_YX', 'Coolant Inlet Temperature', 'Degrees'],
		['TT0651_MA_Y', 'Coolant Outlet Temperature A', 'Degrees'],
		['TT0652_MA_Y', 'Coolant Outlet Temperature B', 'Degrees'],
		['TT0653_MA_Y', 'Coolant Outlet Temperature C', 'Degrees'],
		['TIC0101_CA_Y', 'Coolant Valve Opening A', '%'],
		['TIC0103_CA_Y', 'Coolant Valve Opening B', '%'],
		['TIC0105_CA_Y', 'Coolant Valve Opening C', '%'],
	]

def getConfigG_columns():
	return [
		['PDI0064', 'Process Pressure Difference', 'Bar'],
		['TI0066', 'Process Outlet Temperature','Degrees'],
		['TZI0012', 'Process Inlet Temperature', 'Degrees'],
		['FI0010', 'Process Flow Rate', 'M^3/day'],
		['TT0025', 'Coolant Inlet Temperature', 'Degrees'],
		['TT0026', 'Coolant Outlet Temperature', 'Degrees'],
		['PI0001', 'Coolant Inlet Pressure', 'Bar'],
		['FI0027', 'Coolant Flow Rate', 'M^3/day'],
		['TIC0022U', 'Coolant Valve Opening', '%'],
		['PDT0024', 'Coolant Pressure Difference', 'Bar'],
	]