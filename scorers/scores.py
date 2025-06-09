from braintrust import projects
from autoevals.llm import LLMClassifier
from helpers.helpers import get_project
from pydantic import BaseModel

class EvalCaseExample(BaseModel):
    input: str
    output: str
    expected: str = None

project = get_project()

forgetfulness = LLMClassifier(
        name="forgetfulness",
        model="gpt-4o-mini",
        max_tokens=15000,
        prompt_template="""
You are an expert evaluator tasked with assessing a chatbot's forgetfulness during multi-turn conversations. 

DEFINITION OF FORGETFULNESS:
A chatbot exhibits forgetfulness when it fails to remember and appropriately use information from earlier parts of the conversation that should inform its current response.

EVALUATION CRITERIA:
Assess the chatbot's performance across these dimensions:

1. **Context Retention**: Does the chatbot remember key facts, preferences, or details mentioned earlier?
2. **Conversation Flow**: Does the chatbot maintain logical continuity with previous exchanges?
3. **User Information**: Does the chatbot remember user-specific information (name, preferences, previous requests)?
4. **Task Continuity**: If working on a multi-step task, does the chatbot remember the overall goal and progress?
5. **Reference Resolution**: Can the chatbot correctly interpret pronouns and references to earlier topics?

SCORING GUIDELINES of forgetfulness:
- **NOT (Score: 1)**: The chatbot demonstrates excellent memory, referencing earlier conversation appropriately, maintaining context, and building upon previous exchanges naturally.
- **MILDLY (Score: 0.7)**: The chatbot mostly maintains context but occasionally misses minor details or makes small inconsistencies.
- **MODERATELY (Score: 0.4)**: The chatbot shows clear gaps in memory, failing to connect current responses to earlier context in important ways.
- **HIGHLY (Score: 0)**: The chatbot frequently ignores or contradicts earlier information, treats each message as isolated, or fails to maintain basic conversation continuity.

SPECIAL CONSIDERATIONS:
- New topics or context switches are acceptable and should not be penalized
- Technical limitations (e.g., context window limits) should be considered
- Focus on information that reasonably should be remembered within the conversation scope
- Consider whether the forgetfulness impacts the user experience significantly

INSTRUCTIONS:
1. Carefully read through the entire conversation
2. Identify key information that the chatbot should reasonably remember
3. Look for instances where the chatbot either successfully uses or fails to use prior context
4. Assign a score based on the guidelines above
5. Provide a brief explanation (2-3 sentences) justifying your score

<CONVERSATION>
{{input}}
</CONVERSATION>

<RESPONSE>
{{output}}
</RESPONSE>

FORMAT YOUR RESPONSE AS:
{choice: "HIGHLY", "MODERATELY", "MILDLY", or "NOT", rationale: "Your reasoning for the score, highlighting specific examples of memory retention or forgetfulness"}

        """,
        choice_scores={"HIGHLY": 0, "MODERATELY": 0.4, "MILDLY": 0.7, "NOT": 1},
        
    )

def forgetfulness_handler(input, output):
    return forgetfulness.eval(
        input=input,
        output=output,
    )

project.scorers.create(
    name="forgetfulness",
    slug="forgetfulness",
    description="Evaluate the forgetfulness of a chatbot",
    handler=forgetfulness_handler,
    parameters=EvalCaseExample
)