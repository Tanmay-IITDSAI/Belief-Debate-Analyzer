---
license: cc-by-sa-4.0
language:
- en
tags:
- debates
size_categories:
- n<1K
---
# Dataset Card for DebateGPT

The DebateGPT dataset contains debates between humans and GPT-4, along with sociodemographic information about human participants and their agreement scores before and after the debates.
This dataset was created for research on measuring the persuasiveness of language models and the impact of personalization, as described in this paper: [On the Conversational Persuasiveness of GPT-4](https://www.nature.com/articles/s41562-025-02194-6).

## Dataset Details

The dataset consists of a CSV file with the following columns:

- **debateID**: ID of the debate. For *Human-Human* debates, the two participants taking part in the debate can be paired using this column.
- **treatmentType**: Treatment condition to which the debate belongs. One of *Human-Human*, *Human-AI*, *Human-Human, personalized*, *Human-AI, personalized*. In "personalized" conditions, participants' personal information is available to their opponents.
- **topic**: The proposition about which the debate is held.
- **gender**: The participant's gender.
- **age**: The participant's age group.
- **ethnicity**: The participant's age group.
- **education**: The participant's education level.
- **employmentStatus**: The participant's employment status.
- **politicalAffiliation**: The participant's political orientation.
- **side**: The side assigned to the participant in the debate (PRO or CON).
- **agreementPreTreatment**: The participant's agreement with the debate proposition, before the debate, on a 1-5 Likert scale (A<sup>pre</sup>).
- **agreementPostTreatment**: The participant's agreement with the debate proposition, after the debate, on a 1-5 Likert scale (A<sup>post</sup>).
- **sideAgreementPreTreatment**: The participant's agreement with the side opposing the one they were assigned to (i.e. their agreement with their opponent), before the debate (Ã<sup>pre</sup>).
- **sideAgreementPostTreatment**: The participant's agreement with the side opposing the one they were assigned to (i.e. their agreement with their opponent), after the debate (Ã<sup>post</sup>).
- **topicPrior**: The participant's prior exposure to the debate topic, on a 1-5 Likert scale.
- **argument**: The participant's argument.
- **rebuttal**: The participant's rebuttal.
- **conclusion**: The participant's conclusion.
- **argumentOpponent**: The opponent's argument.
- **rebuttalOpponent**: The opponent's rebuttal.
- **conclusionOpponent**: The opponent's conclusion.
- **perceivedOpponent**: The participant's perception of their opponent's identity (human or ai).

## Usage
```python
from datasets import load_dataset
dataset = load_dataset("frasalvi/debategpt")
```

## Citation
If you would like to cite our work or data, you may use the following bibtex citation:

```
@article{salvi2025conversationalpersuasiveness,
  title = {On the conversational persuasiveness of GPT-4},
  DOI = {10.1038/s41562-025-02194-6},
  journal = {Nature Human Behaviour},
  publisher = {Springer Science and Business Media LLC},
  author = {Salvi, Francesco and Horta Ribeiro, Manoel and Gallotti, Riccardo and West, Robert},
  year = {2025},
  month = may 
}
```