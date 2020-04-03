import src.core as mlApi

# 1. Define dataset specifics 

# File path to dataset .csv
filename = "../master-thesis-db/datasets/D/dataC.csv"

# A desired name for the dataset, used as plot titles
datasetName = "D - Simulated"

# List of columns on form ['name', 'desc', 'unit']
columns = [
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

# List of column names to ignore completely
irrelevantColumns = [
    '50FT001',
    '20PT001',
    '50PT001',
    '20PDT001',
    '50PDT001',
]

# List of training periods on form ['start', 'end']
traintime = [
        ["2020-01-01 00:00:00", "2020-04-01 00:00:00"],
    ]

# Entire testing period
testtime = [
    "2020-01-01 00:00:00",
    "2020-08-01 00:00:00"
]

# Part 1 of the testing period
testtime1 = [
    "2020-04-01 00:00:00",
    "2020-05-05 00:00:00"
]

# Part 2 of the testing period
testtime2 = [
    "2020-05-06 00:00:00",
    "2020-08-01 00:00:00",
]

print("Finding PCA plot for dataset "+datasetName)
print(" ")

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
df_test_1, df_test_2 = mlApi.getTestTrainSplit([testtime1], testtime2)

# 3. Plot scatter plot of training data with color scaling
mlApi.pcaPlot(df_train, plotTitle=datasetName)

# 4. Plot scatter plot of testing data
mlApi.pcaDuoPlot(df_train, df_test_1, df_test_2, plotTitle=datasetName)

# Reset to prepare for second dataset
# -------------------------------------
mlApi.reset()
# -------------------------------------

# 1. 

filename = "../master-thesis-db/datasets/F/data2_180min.csv"

datasetName = "F - Real HX"

columns = [
	['FYN0111', 'Gasseksport rate', 'MSm^3/d'],
	['TT0106_MA_Y', 'Varm side C temperatur inn', 'degrees'],
	['TIC0105_CA_YX', 'Varm side C temperatur ut', 'degrees'],
	['TI0115_MA_Y', 'Scrubber temperatur ut', 'degrees'],
	['PIC0104_CA_YX', 'Innløpsseparator trykk', 'Barg'],
	['TIC0425_CA_YX', 'Kald side temperatur inn', 'degrees'],
	['TT0653_MA_Y', 'Kald side C temperatur ut', 'degrees'],
	['TIC0105_CA_Y', 'Kald side C ventilåpning', '%'],
]

irrelevantColumns = [

]


traintime = [
        ["2018-01-01 00:00:00", "2018-08-01 00:00:00"],
    ]

testtime = [
    ["2018-01-01 00:00:00", "2019-05-01 00:00:00"]
]

testtime1 = [
    "2018-09-25 00:00:00",
    "2018-12-10 00:00:00"
]

testtime2 = [
    "2019-02-15 00:00:00",
    "2019-05-01 00:00:00",
]

print("Finding PCA plot for dataset "+datasetName)
print(" ")

# 2. 
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
df_test_1, df_test_2 = mlApi.getTestTrainSplit([testtime1], testtime2)

# 3. 
mlApi.pcaPlot(df_train, plotTitle=datasetName)

# 4. 
mlApi.pcaDuoPlot(df_train, df_test_1, df_test_2, datasetName)

# Reset to prepare for second dataset
# -------------------------------------
mlApi.reset()
# -------------------------------------

# 1. 

filename = "../master-thesis-db/datasets/G/data_180min.csv"

datasetName = "G - Real HX"

columns = [
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

irrelevantColumns = [
    'PDI0064',
    'PDT0024',
    'FI0027',
    'PI0001',
]

traintime = [
	["2019-04-10 00:00:00", "2019-08-01 00:00:00"]
]

testtime = [
	"2017-01-01 00:00:00",
	"2020-03-01 00:00:00",
]


testtime1 = [
    "2019-02-01 00:00:00",
    "2019-04-01 00:00:00",
]

testtime2 = [
    "2019-09-01 00:00:00",
    "2019-11-10 00:00:00",
]

print("Finding PCA plot for dataset "+datasetName)
print(" ")

# 2. 
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
df_test_1, df_test_2 = mlApi.getTestTrainSplit([testtime1], testtime2)

# 3. 
mlApi.pcaPlot(df_train, plotTitle=datasetName)

# 4. 
mlApi.pcaDuoPlot(df_train, df_test_1, df_test_2, datasetName)

