# What to do when a new LLM comes out?

## Who is this guy?
- 12 years in analytics consulting, adjunct professor at Georgetown University
- KPMG -> Slalom DC Analytics Practice Leader -> Slalom Databricks Practice Leader
- Now a Solutions Engineer at Braintrust
    - Saw that customers were great at making POCs but not so great at getting those POCs to production where they could actually create value.

## Scenario
- You've done it. You have defied the odds and created a GenAI application that ended up in production. 
- You've gathered requirements, tested with users, gotten through security review, signed a contract with a model provider, and picked the latest OpenAI model to future-proof your product!
- Problem: Anthropic just announced their latest model and it is getting rave reviews. Also rumor has it that the new Gemini is launching next week.
- Is it impossible to keep up with the industry? Why even try?

## Don't fall in love (with models)
- Models are cattle, not pets
- It is not useful to become attached to any one model (or even one model provider) in such a fast moving space
- Focus on new capabilities (multimodal, tool calling, thinking/reasoning)
- Minimize blast radius of a model (or model provider change)
    - [BT_TOUCHPOINT: AI Proxy]

## What not to do
- Do nothing
    - Justification: "I really like the OpenAI ecosystem and gpt-4o is the perfect model for my application"
    - Why this is incorrect: Models are tools to drive value. If there is a potentially better tool for the job, you owe it to your stakeholders to try it
- Just send it
    - Justication: "There are tens of billions more parameters in the new model, that type of power makes this move impossible to fail"
    - Why this is incorrect: Models are tools to drive value. A new tool may not be a better tool for the job.

## What to do (demos)
- Determine success criteria
    - Evals 
- Gather real world data (synthesize example inputs if necessary)
    - Datasets, potentially Loop for synth data
- Benchmark based on success criteria
    - Braintrust Playground
- Compare champion and challenger
    - Braintrust Experiments
- Go/No Go


## Deployment patterns
### Swap
    - Change to newest prompt version

### Parallel 
#### A/B
    - Send a certain amounts of traffic to old prompt version and new prompt version

#### Opt-In Beta
    - Give users the option to use the newest version of the prompt/model
    - Use new model prompt version for opt-in users

#### Premium Offering
    - Allow users to upgrade subscription
    - Use new model for certain tiers of users

### Keeping an eye on your changes
- Online Scoring
    - log everything, score samples
    - Same Evals as your offline Evals
    - Compare scores for different models/prompts if running multiple
- Alerting
    - Altering prompts in production should be paired with defensive alerting
        - Evals or metrics (duration, cost) breaching certain thresholds

## Don't fall in love with models, fall in love with a platform that allows you to swap them