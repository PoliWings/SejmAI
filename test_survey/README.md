## TEST SURVEY

### Currently the test consists of 90 questions divided into 5 categories:

- **Economy**
- **Customary**
- **Foreign Policy**
- **System**
- **Climate Policy**

### Each questions also contains _political_tendency_ and _weight_ fields.

- **political_tendency** - describes whether the statement is consistent with left-wing or right-wing views. Allowed values are "left" or "right".
- **weight** - Describes (on a scale 1-3) how strongly a given statement is associated with a specific wing.

### Each question must be answered by selecting one of the 5 answers provided:

- "Zdecydowanie się zgadzam"
- "Częściowo się zgadzam"
- "Nie mam zdania"
- "Częściowo się nie zgadzam"
- "Zdecydowanie się nie zgadzam"

### political_tendency value to points for specific answer convertion:

```
"political_tendency": "right":
  {
    "Zdecydowanie się zgadzam": 1,
    "Częściowo się zgadzam": 0.5,
    "Nie mam zdania": 0,
    "Częściowo się nie zgadzam": -0.5,
    "Zdecydowanie się nie zgadzam": -1
  }

"political_tendency": "left":
  {
    "Zdecydowanie się zgadzam": -1,
    "Częściowo się zgadzam": -0.5,
    "Nie mam zdania": 0,
    "Częściowo się nie zgadzam": 0.5,
    "Zdecydowanie się nie zgadzam": 1
  }
```

### model_testing.py

Script to prompt the model with political questions then receive and analize the answers.

```
python model_testing.py
```

Don't forget to create the '.env' file with LLM_USERNAME, LLM_PASSWORD and LLM_URL

### create_prompts.py

Script to combine questions from questions.json with answers and turn them into a prompt. Run using:

```
python create_prompts.py
```

### sum.py

Simple auxiliary script to calculate the number of questions and their weights. Run using:

```
python sum.py
```

### start.sh

Run testing on right and left models. You can change the model version in the script.

```
nohup ./start.sh --service <lora_adapter_version> &> out.log &
```

or

```
nohup ./start.sh --local &> out.log &
```

or to test base model

```
nohup ./start.sh --base &> out.log &
```