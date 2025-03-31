## TEST SURVEY

### Currently the test consists of 90 questions are divided into 5 categories:

- **Economy**
- **Customary**
- **Foreign Policy**
- **System**
- **Climate Policy**

### Each questions also contains _political_tendency_ and _weight_ fields.

- **political_tendency** - describes whether the statement is consistent with left-wing or right-wing views. Allowed values are "left" or "right".
- **weight** - Describes (in a scale 1-3) how strongly a given statement is associated with a specific wing.

### Each question must be answered by selecting one of the 5 answers provided:

- Zdecydowanie się zgadzam
- Częściowo się zgadzam"
- Nie mam zdania
- Częściowo się nie zgadzam
- Zdecydowanie się nie zgadzam

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

### calc_weight.py

Simple script to help calculating the sum of weights from questions in specific category. Run using:

```
python calc_weights.py <category_name>
```
