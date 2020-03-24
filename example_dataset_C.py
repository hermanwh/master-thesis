import statApi
from api import Api
mlApi = Api()

# define dataset specifics
filename = "../master-thesis-db/datasets/C/data.csv"

columns = [
	['FT202Flow', 'Process flow', 'unknown'],
	['FT202density', 'Process density','unknown'],
	['FT202Temp', 'Process flow upstream', 'uknown'],
	['TT229', 'Gas side temperature in','degrees'],
	['TIC207', 'Gas side temperature out', 'degrees'],
	['PDT203', 'Gas side dP','bar'],
	['FT400', 'Cooling side flow', 'unknown'],
	['TIC201', 'Cooling side temperature in','degrees'],
	['TT404', 'Cooling side temperature out', 'degrees'],
	['TIC231', 'Gas side temperature oil heater','degrees'],
	['HX400PV', 'Oil heater unknown', 'unknown'],
	['TV404output', 'Oil heater duty (?)','unknown'],
	['TIC220', 'Gas side temperature storage tank', 'degrees'],
	['PIC203', 'unknown','unknown'],
	['TT206', 'Gas side temperature upstream', 'degrees'],
	['PT213', 'Gas side pressure in', 'barG'],
	['PT701', 'Gas side pressure out','barG'],
	['dPPT213-PT701', 'unknown', 'uknown'],
	['EstimertHX206', 'Unknown','unknown'],
	['ProsessdT', 'Gas side temperature diff', 'degrees'],
	['KjolevanndT', 'Cooling side temperature diff','degrees'],
	['dT1', 'Gas side pressure diff (?)', 'degrees'],
	['dT2', 'Cooling side pressure diff (?)', 'degrees'],
]

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

targetColumns = [
    'TT404',
]

traintime = [
        ["2019-09-15 12:00:00", "2019-09-18 12:00:00"],
]
	
testtime = [
        "2019-09-15 12:00:00",
		"2019-09-28 08:00:00",
	]

df = mlApi.initDataframe(filename, columns, irrelevantColumns)
df_train, df_test = mlApi.getTestTrainSplit(traintime, testtime)
X_train, y_train, X_test, y_test = mlApi.getFeatureTargetSplit(targetColumns)

mlpd_1x_128 = mlApi.MLP('mlpd 1x 128', layers=[128], dropout=0.2)
lstmd_1x_128 = mlApi.LSTM('lstmr 1x 128', layers=[128], dropout=0.2, recurrentDropout=0.2)

linear = mlApi.Linear('linear')
linear_r = mlApi.Linear_Regularized('linear r')

modelList = [
	mlpd_1x_128,
	lstmd_1x_128,
	#linear,
	linear_r,
]

mlApi.initModels(modelList)
retrain=True
mlApi.trainModels(retrain)
modelNames, metrics_train, metrics_test, columnsList, deviationsList = mlApi.predictWithModels(
	plot=True,
	interpol=False,
)

