from sklearn import preprocessing

# 数据标准化
preprocessing.scale(dt)  # 标准化数据

scaler = preprocessing.StandardScaler().fit(dt)
scaler.transform([np.random.rand(5)])

# 0-1 标准化
min_max_scaler=preprocessing.MinMaxScaler()
scaler=min_max_scaler.fit_transform(df) 
min_max_scaler.transform([np.random.rand(5)])