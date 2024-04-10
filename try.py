metrics = ['mse', 'rmse', 'mae', 'r2']
total_metrics = []
model_metric = {}
a = 1
b = 2
c = 3
d = 4
model_metric = {'mse':a, 'rmse':b, 'mae':c, 'r2':d}
total_metrics.append(model_metric)
model_metric = {'mse':a+3, 'rmse':b+2, 'mae':c+4, 'r2':d+1}
total_metrics.append(model_metric)



for model_metric in total_metrics:
    mse = model_metric['mse']
    rmse = model_metric['rmse']
    mae = model_metric['mae']
    r2 = model_metric['r2']
    print(f'mse: {mse}, rmse: {rmse}, mae: {mae}, r2: {r2}')