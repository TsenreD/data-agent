### Objective
Build an agent for smart data selection (Active Learning) or assembling a multimodal dataset. Students must choose one track. The agent will integrate into the final pipeline as `active_learning_op`.

### Track A: ActiveLearningAgent

### Architecture
- Implement the core `ActiveLearningAgent` class.
- Use `fit(labeled_df) -> model` to train the base model.
- Use `query(pool, strategy) -> indices` to select samples using strategies like entropy, margin, or random.
- Use `evaluate(labeled_df, test_df) -> Metrics` to calculate metrics such as accuracy and F1 score.
- Use `report(history) -> LearningCurve` to generate a plot of quality versus the number of labeled examples.

### Technical Contract
```python
from agents.active_learning import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')

# Cycle: start with N=50, 5 iterations of 20 examples
history = agent.run_cycle(
    labeled_df=df_labeled_50, 
    pool_df=df_unlabeled, 
    strategy='entropy', 
    n_iterations=5, 
    batch_size=20
)

# -> history: list of {iteration, n_labeled, accuracy, f1}

agent.report(history) # -> learning_curve.png
```

### Deliverables
- Execute the active learning cycle starting with N=50 for 5 iterations to produce the final model.
- Plot learning curves comparing entropy versus random selection strategies on a single graph.
- Provide a conclusion analyzing how many training examples were saved to achieve the same quality compared to a random baseline.
- Submit the files `agents/al_agent.py` and `notebooks/al_experiment.ipynb`.
