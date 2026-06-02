"""Prompt templates for the evolutionary loop.

The crossover / mutation / critic prompts are transcribed verbatim from the
paper's Appendix D. The objective prompt is a reconstruction: §4.2 only states
the description-bearing phrasing ("Given a satellite image, write a function to
estimate <descr>"); the exact API-injection format is not printed in the paper,
so the wrapper below is a reasonable reconstruction that also surfaces the
primitive API spec the LLM must compose against.

Placeholders are Python ``str.format`` fields filled by the loop at runtime.
"""

# Objective prompt (§4.2): used for initialization and embedded as context in
# crossover/mutation. {descr} = task description; {api_spec} = primitive docstrings.
OBJECTIVE_PROMPT = """Given a satellite image, write a function to estimate {descr}.

You have access to the following functions:
{api_spec}

Write a Python function called `estimator` that uses these functions to compute useful features.
Return a flat tuple of individually named scalar features (e.g. `return feature1, feature2, feature3`). Do not use list comprehensions, dictionary unpacking, starred expressions, or loops in the return statement.
Only give me the code."""

# Crossover prompt (Appendix D, verbatim).
CROSSOVER_PROMPT = """{program1}
This program has a score of {score1}.
{program2}
This program has a score of {score2}.
Can you write a function that gives a higher score? Feel free to combine elements that worked from both programs. Only give me the code."""

# Mutation prompt (Appendix D, verbatim).
MUTATION_PROMPT = """{program}
This program has a score of {score}.
Can you edit this code to write a better function for the problem?
Only give me code."""

# Critic prompt (Appendix D, verbatim). {bad_categories} = comma-separated
# concept names the program performs worst on.
CRITIC_PROMPT = """The program you generated is bad when the satellite image contains the following categories: {bad_categories}. Generate a program that also works on these categories."""

# "No problem context" ablation objective prompt (Appendix D, verbatim).
NO_CONTEXT_OBJECTIVE_PROMPT = """Write a function that takes in an image and returns a useful map using it."""

# NOTE on scoring direction (resolved in Step 5):
# Appendix D phrases crossover as asking for "a higher score", but our fitness
# is an ERROR (lower = better). The LLM does not compute on the score; it uses it
# as a relative signal. Step 5 will format the score so "higher score" stays
# consistent with "better" (e.g. by passing a negated/inverted error), keeping
# the prompt text verbatim. Left here as a flag for that wiring.

# Natural-language task descriptions for the objective prompt's {descr} field.
_TASK_DESCRIPTIONS = {
    "population_density": "population density",
    "poverty": "the poverty (wealth) index",
    "agb": "aboveground biomass",
}


def get_task_description(benchmark_name: str) -> str:
    """Return the natural-language task description for a benchmark."""
    return _TASK_DESCRIPTIONS.get(benchmark_name, benchmark_name.replace("_", " "))
