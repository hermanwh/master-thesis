def getConfigs():
	return {
		'A': getConfigA,
		'B': getConfigB,
		'C': getConfigC,
		'D': getConfigD,
		'E': getConfigE,
		'F': getConfigF,
	}

def getConfigDirs():
	return getConfigs().keys()

def getConfig(name):
	configs = getConfigs()
	if name in getConfigDirs():
		return configs[name]()
	else:
		return [None, None, None, None, None]

def getConfigA():
	columnDescriptions = {
	    'TIT139.PV': 'Gas side inlet temperature',
	    'TIC205.PV': 'Gas side outlet temperature',
	    'FI125B.PV': 'Gas side flow',
	    'PT140.PV': 'Gas side compressor pressure',
	    'TT069.PV': 'Cooling side inlet temperature',
	    'PT074.PV': 'Cooling side pressure',
	    'TI205.OUT': 'Cooling side vavle opening',
	    'XV127.CMD': 'Anti-surge compressor valve',
	    'XV127.ZSH': 'Anti-surge valve',
	    'ZT127.PV': 'Anti-surge unknown',
	}

	irrelevantColumns = [
		'XV127.CMD',
		'XV127.ZSH'
	]

	columns = list(columnDescriptions.keys())
	relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))

	columnUnits = None

	traintime = ["2016-07-01 00:00:00", "2016-10-06 00:00:00"]
	testtime = ["2016-10-06 00:00:00", "2017-03-01 00:00:00"]
	validtime = ["2016-09-01 00:00:00", "2016-09-22 00:00:00"]

	timestamps = [traintime, testtime, validtime]

	return [columns, relevantColumns, columnDescriptions, columnUnits, timestamps]

def getConfigB():
	columnDescriptions = {
	    'TT181.PV': 'Gas side inlet temperature',
	    'TIC215.PV': 'Gas side outlet temperature',
	    'FI165B.PV': 'Gas side flow',
	    'PT180.PV': 'Gas side compressor pressure',
	    'TT069.PV': 'Cooling side inlet temperature',
	    'PT074.PV': 'Cooling side pressure',
	    'TIC215.OUT': 'Cooling side vavle opening',
	    'XV167.CMD': 'Anti-surge compressor valve',
	    'XV167.ZSH': 'Anti-surge valve',
	    'ZT167.PV': 'Anti-surge unknown',
	}

	irrelevantColumns = [
		'XV167.CMD',
		'XV167.ZSH'
	]

	columns = list(columnDescriptions.keys())
	relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))

	columnUnits = None

	traintime = ["2016-07-01 00:00:00", "2016-10-06 00:00:00"]
	testtime = ["2016-07-01 00:00:00", "2017-03-01 00:00:00"]
	validtime = ["2016-09-01 00:00:00", "2016-09-22 00:00:00"]

	timestamps = [traintime, testtime, validtime]

	return [columns, relevantColumns, columnDescriptions, columnUnits, timestamps]


def getConfigC():
	columnDescriptions = {
	    "Date": "Date",
	    "FT202Flow": "Process flow",
	    "FT202density": "Process density",
	    "FT202Temp": "Process flow upstream",
	    "TT229": "Process temperture in HX",
	    "TIC207": "Process temperature out HX",
	    "PDT203": "Process dP HX",
	    "FT400": "Cooling flow",
	    "TIC201": "Cooling temperature in HX",
	    "TT404": "Cooling temperature out HX",
	    "TIC231": "Process temperature out oil heater",
	    "HX400PV": "Oil heater unknown",
	    "TV404output": "Oil heater unknown (duty?)",
	    "TIC220": "Process temperature tank",
	    "PIC203": "Uknown",
	    "TT206": "Process temperature upstream",
	    "PT213": "Process pressure in HX",
	    "PT701": "Process pressure out HX",
	    "dPPT213-PT701": "Unknown",
	    "EstimertHX206": "Unknown",
	    "ProsessdT": "Process dT",
	    "KjolevanndT": "Cooling dT",
	    "dT1": "Uknown",
	    "dT2": "Unknown",
	}

	irrelevantColumns = [
	    "FT202density",
	    "FT202Temp",
	    "TIC231",
	    "HX400PV",
	    "TV404output",
	    "TIC220",
	    "PIC203",
	    "TT206",
	    "PT213",
	    "PT701",
	    "dPPT213-PT701",
	    "EstimertHX206",
	    "ProsessdT",
	    "KjolevanndT",
	    "dT1",
	    "dT2",
	]

	columns = list(columnDescriptions.keys())
	relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))

	columnUnits = None

	traintime = ["2019-09-15 12:00:00", "2019-09-18 12:00:00"]
	testtime = ["2019-09-15 12:00:00", "2019-09-28 08:00:00"]
	validtime = ["2019-09-17 12:00:00", "2019-09-18 12:00:00"]
	
	timestamps = [traintime, testtime, validtime]

	return [columns, relevantColumns, columnDescriptions, columnUnits, timestamps]

def getConfigD():
	columnDescriptions = {
	    '20TT001': 'Gas side inlet temperature',
	    '20PT001': 'Gas side inlet pressure',
	    '20FT001': 'Gas side flow',
	    '20TT002': 'Gas side outlet temperature',
	    '20PDT001': 'Gas side pressure differential',
	    '50TT001': 'Cooling side inlet temperature',
	    '50PT001': 'Cooling side inlet pressure',
	    '50FT001': 'Cooling side flow',
	    '50TT002': 'Cooling side outlet temperature',
	    '50PDT001': 'Cooling side pressure differential',
	    '50TV001': 'CM Valve opening'
	}

	irrelevantColumns = [
		'50FT001',
		'50PDT001',
		'20PDT001',
		'50TV001',
	]

	columns = list(columnDescriptions.keys())
	relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))

	columnUnits = None

	traintime = ["2020-01-01 00:00:00", "2020-03-01 00:00:00"]
	testtime = ["2020-01-01 00:00:00", "2020-07-01 00:00:00"]
	validtime = ["2020-03-01 00:00:00", "2020-04-01 00:00:00"]

	timestamps = [traintime, testtime, validtime]

	return [columns, relevantColumns, columnDescriptions, columnUnits, timestamps]

def getConfigE():
	columnDescriptions = {
		'Date': 'Datetime',
		'TT0102': 'Varm side temperatur inn',
		'TT0107': 'Varm side temperatur ut',
		'FT0005': 'Varm side gasseksport',
		'FT0161': 'Varm side veske ut av scrubber',
		'PT0106': 'Varm side trykk innløpsseparator',
		'PDT0105': 'Varm side trykkfall',
		'TT0312': 'Kald side temperatur inn',
		'TT0601': 'Kald side temperatur ut',
		'FT0605': 'Kald side kjølemedium',
		'PDT0604': 'Kald side trykkfall',
		'TIC0108': 'Kald side ventilåpning',
		'HV0175': 'Pumpe bypass',
		'PT0160': 'Pumpe trykk ut',
	}

	irrelevantColumns = [
	]

	columns = list(columnDescriptions.keys())
	relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))

	columnUnits = {
		'TT0102': 'degrees',
		'TT0107': 'degrees',
		'FT0005': 'Sm^3 / h',
		'FT0161': 'Am^3 / h',
		'PT0106': 'BarA',
		'PDT0105': 'BarG',
		'TT0312': 'degrees',
		'TT0601': 'degrees',
		'FT0605': 'Am^3 / h',
		'PDT0604': 'Bar',
		'TIC0108': '%',
		'HV0175': 'bool',
		'PT0160': 'Bar',
	}

	traintime = ["2015-03-01 00:00:00", "2016-01-01 00:00:00"]
	testtime = ["2017-11-13 00:00:00", "2018-06-01 00:00:00"]
	validtime = ["2015-10-01 00:00:00", "2016-01-01 00:00:00"]

	timestamps = [traintime, testtime, validtime]

	return [columns, relevantColumns, columnDescriptions, columnUnits, timestamps]
	
def getConfigF():
	columnDescriptions = {
		'Date':'Date',
		'FYN0111': 'Gasseksport rate',
		'FT0111': 'Gasseksport molvekt',
		'TT0102_MA_Y': 'Varm side A temperatur inn',
		'TIC0101_CA_YX': 'Varm side A temperatur ut',
		'TT0104_MA_Y': 'Varm side B temperatur inn',
		'TIC0103_CA_YX': 'Varm side B temperatur ut',
		'TT0106_MA_Y': 'Varm side C temperatur inn',
		'TIC0105_CA_YX': 'Varm side C temperatur ut',
		'TI0115_MA_Y': 'Scrubber temperatur ut',
		'PDT0108_MA_Y': 'Varm side A trykkfall',
		'PDT0119_MA_Y': 'Varm side B trykkfall',
		'PDT0118_MA_Y': 'Varm side C trykkfall',
		'PIC0104_CA_YX': 'Innløpsseparator trykk',
		'TIC0425_CA_YX': 'Kald side temperatur inn',
		'TT0651_MA_Y': 'Kald side A temperatur ut',
		'TT0652_MA_Y': 'Kald side B temperatur ut',
		'TT0653_MA_Y': 'Kald side C temperatur ut',
		'TIC0101_CA_Y': 'Kald side A ventilåpning',
		'TIC0103_CA_Y': 'Kald side B ventilåpning',
		'TIC0105_CA_Y': 'Kald side C ventilåpning',
	}

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
		'TIC0105_CA_Y',
	]

	columns = list(columnDescriptions.keys())
	relevantColumns = list(filter((lambda column: column not in irrelevantColumns), columns))

	columnUnits = {
		'Date':'Date',
		'FYN0111': 'MSm^3/d',
		'FT0111': 'g/mole',
		'TT0102_MA_Y': 'degrees',
		'TIC0101_CA_YX': 'degrees',
		'TT0104_MA_Y': 'degrees',
		'TIC0103_CA_YX': 'degrees',
		'TT0106_MA_Y': 'degrees',
		'TIC0105_CA_YX': 'degrees',
		'TI0115_MA_Y': 'degrees',
		'PDT0108_MA_Y': 'Bar',
		'PDT0119_MA_Y': 'Bar',
		'PDT0118_MA_Y': 'Bar',
		'PIC0104_CA_YX': 'Barg',
		'TIC0425_CA_YX': 'degrees',
		'TT0651_MA_Y': 'degrees',
		'TT0652_MA_Y': 'degrees',
		'TT0653_MA_Y': 'degrees',
		'TIC0101_CA_Y': '%',
		'TIC0103_CA_Y': '%',
		'TIC0105_CA_Y': '%',
	}
	"""
	traintime = ["2017-07-01 00:00:00", "2018-05-01 00:00:00"]
	testtime = ["2017-07-01 00:00:00", "2020-02-01 00:00:00"]
	validtime = ["2018-01-01 00:00:00", "2018-05-01 00:00:00"]
	
	traintime = ["2017-08-05 00:00:00", "2018-05-01 00:00:00"]
	testtime = ["2017-08-05 00:00:00", "2020-02-01 00:00:00"]
	validtime = ["2018-01-01 00:00:00", "2018-05-01 00:00:00"]
	"""
	traintime = ["2017-08-05 00:00:00", "2018-08-01 00:00:00"]
	testtime = ["2017-08-05 00:00:00", "2019-11-01 00:00:00"]
	validtime = ["2018-05-01 00:00:00", "2018-07-01 00:00:00"]
	"""
	traintime = ["2019-04-15 00:00:00", "2019-06-01 00:00:00"]
	testtime = ["2019-04-15 00:00:00", "2020-02-01 00:00:00"]
	validtime = ["2019-05-01 00:00:00", "2019-07-01 00:00:00"]
	"""
	timestamps = [traintime, testtime, validtime]

	return [columns, relevantColumns, columnDescriptions, columnUnits, timestamps]