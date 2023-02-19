import gpt2
g = gpt2.gpt2()
g.models_dir = r"C:\Users\Tim\Downloads\gpt-2\models"
g.model_name = "1558M"
g.length = 200
g.temperature = 0.8
g.load()

response = g.prompt("")

print(response)
