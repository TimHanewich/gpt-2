I wrote the module `gpt` found [here](./src/gpt2.py). The class `gpt` in this model abstracts GPT-2's capabilities into an easy to use class. An example using it:

```
import gpt2

gen = gpt2.gpt2()
gen.models_dir = r"C:\Users\Tim\Downloads\gpt-2\models"
gen.model_name = "345M"
gen.load()

response = gen.prompt("A woman wakes up in a conference room with no memory of who or where she is. After being given a survey and learning she is Helly, a new hire at Lumon Industries, she is allowed to leave but finds she is unable to do so. She then sees a video explaining that she has undergone the severance procedure, which split her memories to create a version of herself that will only exist inside the workplace. Mark Scout, who works alongside Helly in Lumon's Macrodata Refinement (MDR) division, discovers he is being promoted to department head in light of coworker Petey's sudden departure. The outside version of Mark, a former history professor grieving his wife's death and living in the Lumon-subsidized town of Kier, encounters a man claiming to be Petey who gives him a letter with cryptic instructions. Mark returns home and interacts with his neighbor Mrs. Selvig, unaware that she is his boss, senior manager Harmony Cobel. Helly subjected herself to the severance procedure because ")

print(response)

```

I also wrote the [stripped.py](./src/stripped.py) to demonstrate using the core model (abstracted from [interactive_conditional_samples.py](./src/interactive_conditional_samples.py)).


## Downloading Packages
You'll have to create a virtual environment with the necessary dependencies to run the code here. See steps 2 and 3 from [here](https://timhanewich.medium.com/running-openais-gpt-2-language-model-on-your-pc-5d5e1b9fbb8b) to see how to do this.