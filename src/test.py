f = open(r"C:\Users\Tim\Downloads\gpt-2\src\prompt.txt")
content = f.read()

import gpt2
g = gpt2.gpt2()
g.models_dir = r"C:\Users\Tim\Downloads\gpt-2\models"
g.model_name = "345M"
g.load()

response = g.prompt(content)

print("----------------")
print(content)
print("================")
print(response)
