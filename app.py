# A gist for using the `llama.cpp` model with the `dspy` library.
#
# DSPy features used in this gist
# - `dspy.Predict`
# - `dspy.Signature`
# - `dspy.context`
# - `dspy.evaluate.Evaluate`
#
# The script first prompts the model to answer a example question and assess the correctness and engagingness of the answer using a evaluator.
#
# Install `llama.cpp` from brew with built-in OpenAI-compatible server.
# brew install ggerganov/ggerganov/llama.cpp
# llama-server --hf-repo TheBloke/Mistral-7B-Instruct-v0.2-GGUF --model mistral-7b-instruct-v0.2.Q4_K_M.gguf --hf-file mistral-7b-instruct-v0.2.Q4_K_M.gguf

import dspy
from dspy.evaluate import Evaluate

# Optional for displaying the results on stdout as tables
from rich import print
from rich.table import Table

# The `llama.cpp` model
llama_cpp_model = dspy.OpenAI(
    # assume llama-server is running on localhost:8080
    api_base="http://localhost:8080/v1/",
    # placeholder, or it will raise an error
    api_key="none",
    # for some reasons, an error will be raised if set to `text` (llama-server issue?)
    model_type="chat",
    # stop word for mistral-7b-instruct-v0.2
    stop="\n\n",
    # max number of tokens to generate
    max_tokens=250,
)

dspy.settings.configure(lm=llama_cpp_model)

# The example question-answer pairs, we already know the answer and want to access the correctness and engagingness in the evaluator
examples = [
    dspy.Example(
        question="Are both Nehi and Nectar d.o.o. part of the beverage industry?",
        answer="yes",
    ).with_inputs("question"),
    dspy.Example(
        question="Angela Diniz was born in Belo Horizonte which is the capital of which Brazilian state",
        answer="Minas Gerais",
    ).with_inputs("question"),
    dspy.Example(
        question="Which Indian symphonic conductor serves on the Anotonin Dvorak Music Festival named after the world-renowned Czech composer?",
        answer="Debashish Chaudhuri",
    ).with_inputs("question"),
]

# A dspy signature for automatic assessments of a question-answer pair
class Assess(dspy.Signature):
    """Assess the quality of a answer of a question."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


# the predict module built from the assessment signature
# use in the correct_engaging_metric function
assess_pred = dspy.Predict(Assess)

# a metric returning a score between 0 and 1 for the correctness of the answer and the engagingness of the answer
def correct_engaging_metric(gold, pred, trace=None):
    question, answer, gen_answer = gold.question, gold.answer, pred.answer

    engaging = "Is the assessed text self-contained, information?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    with dspy.context(lm=llama_cpp_model):
        correct = assess_pred(assessed_text=gen_answer, assessment_question=correct)
        engaging = assess_pred(assessed_text=gen_answer, assessment_question=engaging)

    correct, engaging = [
        "yes" in m.assessment_answer.lower() for m in [correct, engaging]
    ]
    score = correct + engaging
    if trace is not None:
        return score >= 2  # noqa: E701
    return score / 2.0


# A predict module accept a signature (can be string or a `dspy.Signature` class)
# the following are example signature strings
# question -> answer
# sentence -> sentiment
# document -> sunmary
# text -> gist
# long_context -> tldr
# content, question -> answer
# question, choices -> reasoning, selection
#
# example:
# predict_module = dspy.Predict('document -> sunmary')

# a predict module for answering questions
qa_predict_module = dspy.Predict("question -> answer")

# init the evaluator with our dataset and the metric
evalute = Evaluate(
    metric=correct_engaging_metric,
    devset=examples,
    num_threads=4,
    display_progress=True,
)

score = evalute(qa_predict_module)

table = Table(title="Evalute")
table.add_column("Number of Examples")
table.add_column("Evalute Score (%)", style="green")

table.add_row(str(len(examples)), str(score))

print(table)
