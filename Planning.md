I’m reviewing the generator, archive, and instance flow now so I can propose a change that fits the existing loop rather than disrupting it.

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/generators/evolutionary.py#1-1), lines 1 to 260

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/archives/_archive.py)

The core flow is clearer now; I’m tracing where evaluation and descriptor computation split so I can suggest an efficient batch-oriented path that preserves the generator structure.

Searched for regex `def _evaluate_population|descriptor_pipe\(|fitness|descriptor` (`**/digneapy/**/*.py`), 534 results

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/core/_instance.py#1-1), lines 1 to 260

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/generators/_base_generator.py#1-1), lines 1 to 280

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/generators/_base_generator.py#280-280), lines 280 to 311

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/core/_descriptors.py)

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/generators/evolutionary.py#240-240), lines 240 to 360

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/generators/_utils.py)

Searched for regex `def extract_features\(|generate_problems_from_instances\(` (`**/digneapy/**/*.py`), 11 results

Read [](file:///Users/amarrero/projects/DIGNEApy/digneapy/core/_domain.py)

The current structure already supports a batched path, and the cleanest option is to bundle evaluation plus descriptor generation behind a single helper so the outer generator loop stays intact.

Created memory file [](file:///Users/amarrero/Library/Application%20Support/Code/User/workspaceStorage/1f8a288bb9b558f43f4d02ca704dd867/GitHub.copilot-chat/memory-tool/memories/YTc3YTNlMzAtYmY4NC00ZWE5LWEwZmEtY2FmMjQ2NjE1ZGEy/plan.md)

## Analysis

Yes — the design already supports a batch-oriented path, and you do not need to reshape the whole Evolutionary loop to get it.

### What the current code already does
- The base generator in _base_generator.py evaluates an entire population at once through `_evaluate_population`.
- The descriptor pipeline in _descriptors.py also works on a full batch of instances.
- The Evolutionary generator in evolutionary.py currently performs these as two separate steps.
- The Instance class in _instance.py is mostly a container: it stores fitness, descriptor, novelty, portfolio scores, etc., but it does not compute them.

### The efficient approach
The best change is to combine:
1. performance evaluation,
2. descriptor computation,

into a single helper, while keeping the outer generation loop unchanged.

That helper would do roughly this:

- evaluate the batch to obtain performance bias and raw solver scores,
- pass those scores into the descriptor pipeline once,
- return both the performance bias and descriptors together.

This is efficient because the descriptor step can reuse the same portfolio scores that were just computed.

## Important nuance

You cannot fully compute the final blended fitness in the same step as the descriptors, because the final fitness also depends on novelty, and novelty depends on the archive and the descriptors of the current batch. So the ideal split is:

- batch evaluation + descriptor extraction: one shared step,
- novelty computation,
- final fitness computation.

That still keeps the main loop intact.

## Recommended change

The minimal-change refactor would be:

- add a helper such as `_evaluate_and_describe_population` in _base_generator.py,
- use it in evolutionary.py for both the initial population and each offspring batch,
- leave archive updates, novelty scoring, and replacement logic exactly where they are.

## Practical conclusion

Yes, it can be done without altering the main loop of the Evolutionary generator. The change is local and should be implemented at the generator base layer rather than inside the archive or the Instance class.