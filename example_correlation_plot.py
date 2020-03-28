import statApi
import pandas as pd
from api import Api
mlApi = Api()

# define dataset specifics
filename = "../master-thesis-db/datasets/D/dataC.csv"

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

irrelevantColumns = [

]

traintime = [
        ["2020-01-01 00:00:00", "2020-04-01 00:00:00"],
    ]

testtime = [
    "2020-01-01 00:00:00",
    "2020-08-01 00:00:00"
]

testtime1 = [
    "2020-04-15 00:00:00",
    "2020-05-04 00:00:00"
]

testtime2 = [
    "2020-06-01 00:00:00",
    "2020-06-16 00:00:00"
]

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
df_test_1, df_test_2 = mlApi.getTestTrainSplit([testtime1], testtime2)
df_test_joined = pd.concat([df_test_1, df_test_2])

covmat_train = statApi.correlationMatrix(df_train)
covmat_test_1 = statApi.correlationMatrix(df_test_1)
covmat_test_2 = statApi.correlationMatrix(df_test_2)

covmat_diff_1 = covmat_train - covmat_test_1
covmat_diff_2 = covmat_train - covmat_test_2

statApi.correlationPlot(df_train, "D train")
statApi.correlationDuoPlot(df_test_1, df_test_2, "D test 1", "D test 2")
statApi.correlationDifferencePlot(df_train, df_test_joined, "Difference, D train and D test")



# define dataset specifics
filename = "../master-thesis-db/datasets/F/data2_360min.csv"

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

traintime2 = [
	["2019-02-20 00:00:00", "2019-03-20 00:00:00"]
]

testtime = [
    "2018-12-01 00:00:00",
    "2019-02-01 00:00:00"
]

# 2. Initiate and divide data
df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
df_train2, df_test2 = mlApi.getTestTrainSplit(traintime2, testtime)

statApi.correlationDuoPlot(df_train, df_train2, "F train 1", "F train 2")
statApi.correlationPlot(df_test, "F test")
statApi.correlationDifferencePlot(df_train, df_test, "Difference, F train 1 and F test")
statApi.correlationDifferencePlot(df_train2, df_test, "Difference, F train 2 and F test")
statApi.correlationDifferencePlot(df_train, df_train2, "Difference, F train 1 and F train 2")