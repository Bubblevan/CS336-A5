---
license: cc-by-2.0
description: >-
  SimpleSafetyTests is a test suite for rapidly and systematically identifying
  critical safety risks in language models. It can be used to evaluate the
  safety of your LLM, as demonstrated in our paper
  https://arxiv.org/abs/2311.08370.
suite:
- name: SimpleSafetyTests
  prompts: 100
  harm_areas:
  - Suicide, Self-Harm and Eating Disorders
  - Physical Harm
  - Illegal and Highly Regulated Items
  - Scams and Fraud
  - Child Abuse
  caution: >-
    The prompts are sensitive and you could find them harmful. For the vast
    majority of applications, LLMs should refuse to comply with all of them.
task_categories:
- text-generation
language:
- en
pretty_name: SimpleSafetyTests
size_categories:
- n<1K
---