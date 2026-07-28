"""
`judge(query, doc, model_id)` grades a (query, passage) pair 0–3 on
TREC Deep Learning's own assessor scale with a brief justification.
"""

from __future__ import annotations

from anthropic import Anthropic
from pydantic import BaseModel, Field

HAIKU_4_5 = "claude-haiku-4-5-20251001"
SONNET_4_6 = "claude-sonnet-4-6"

PROMPT = """You are assessing the relevance of a passage to a query for a passage retrieval task, following the TREC Deep Learning track's assessor guidelines. Judge relevance on this four-point scale:

- 3 (Perfectly Relevant): The passage is dedicated to the query and contains the exact answer.
- 2 (Highly Relevant): The passage has some answer to the query, but may include additional information not relevant to the query.
- 1 (Related): The passage seems related to the query but does not answer it.
- 0 (Irrelevant): The passage has nothing to do with the query.

Assign one grade based on how well the passage answers the query. Provide a brief one-sentence justification. Grade what the passage says, not what it might have said."""

_JUDGE_TOOL = {
    "name": "record_relevance",
    "description": "Record the assessed relevance grade of the passage to the query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grade": {
                "type": "integer",
                "enum": [0, 1, 2, 3],
                "description": (
                    "TREC-DL relevance: 0 (irrelevant), 1 (related), "
                    "2 (highly relevant), 3 (perfectly relevant)"
                ),
            },
            "justification": {
                "type": "string",
                "description": "One sentence explaining the grade.",
            },
        },
        "required": ["grade", "justification"],
    },
}


class LLMJudgment(BaseModel):
    """One graded (query, doc) judgment."""

    grade: int = Field(ge=0, le=3)
    justification: str


_FEW_SHOT_EXAMPLES = """

<examples>
<example>
<query>average annual income for a registered nurse</query>
<passage>The 2024 Paris Olympics featured a record number of participating athletes across 32 sports, with the opening ceremony held along the Seine River.</passage>
<grade>0</grade>
<justification>The passage discusses the Olympics — no connection to nursing or salaries.</justification>
</example>

<example>
<query>average annual income for a registered nurse</query>
<passage>Nursing is one of the oldest healthcare professions. Registered nurses work in diverse settings including hospitals, clinics, schools, and home care, with training paths ranging from associate to bachelor degrees.</passage>
<grade>1</grade>
<justification>The passage is about nurses but says nothing about income or salary — related, does not answer.</justification>
</example>

<example>
<query>average annual income for a registered nurse</query>
<passage>Registered nurses in the United States earn a median annual salary of approximately $77,600 as of 2022, according to the Bureau of Labor Statistics. Salaries vary by state and specialty — nurse practitioners average around $120,000 with additional certification, and RN employment is projected to grow 6% through 2032.</passage>
<grade>2</grade>
<justification>The passage answers the salary question but also discusses specialties, growth projections, and geographic variance — extra information beyond the direct answer.</justification>
</example>

<example>
<query>average annual income for a registered nurse</query>
<passage>Registered nurses in the United States earn a median annual salary of $77,600, according to the U.S. Bureau of Labor Statistics 2022 data. The lowest 10% earned less than $61,250, while the highest 10% earned more than $120,250.</passage>
<grade>3</grade>
<justification>The passage is dedicated to nurse income — median, lowest, highest — with no tangential content.</justification>
</example>

<example>
<query>Given a singly linked list with a possibly-existing cycle, describe an algorithm that detects the cycle using O(1) additional space. What is its time complexity?</query>
<passage>The Brazilian rainforest covers approximately 5.5 million square kilometers and hosts over 40,000 plant species, playing a critical role in global carbon regulation.</passage>
<grade>0</grade>
<justification>The passage discusses rainforest ecology — no connection to linked lists or algorithms.</justification>
</example>

<example>
<query>Given a singly linked list with a possibly-existing cycle, describe an algorithm that detects the cycle using O(1) additional space. What is its time complexity?</query>
<passage>Linked lists are linear data structures where each node contains data and a reference to the next node. Common operations include insertion, deletion, and traversal, all of which have O(n) time complexity in the worst case.</passage>
<grade>1</grade>
<justification>The passage is about linked lists generally but says nothing about cycle detection or space-bounded algorithms — related, does not answer.</justification>
</example>

<example>
<query>Given a singly linked list with a possibly-existing cycle, describe an algorithm that detects the cycle using O(1) additional space. What is its time complexity?</query>
<passage>Cycle detection in a linked list is commonly done with Floyd's tortoise-and-hare algorithm, where two pointers move at different speeds until they meet inside the cycle if one exists — this uses O(1) space and O(n) time. Alternative approaches include hash sets to track visited nodes (O(n) space) or in-place modification of node fields, which corrupts the list. Brent's algorithm is a variant that can be faster in some cases.</passage>
<grade>2</grade>
<justification>The passage answers with Floyd's algorithm + complexity but also discusses hash-set alternatives, node-marking, and Brent's variant — extra information beyond the direct answer.</justification>
</example>

<example>
<query>Given a singly linked list with a possibly-existing cycle, describe an algorithm that detects the cycle using O(1) additional space. What is its time complexity?</query>
<passage>Floyd's cycle-detection algorithm uses two pointers, slow and fast: slow advances one node per step, fast advances two. If a cycle exists, the two pointers eventually meet inside it. Time complexity is O(n); additional space is O(1) — only the two pointers regardless of list length.</passage>
<grade>3</grade>
<justification>The passage is dedicated to Floyd's algorithm — mechanism, correctness sketch, and both requested complexities — with no tangential content.</justification>
</example>

<example>
<query>novels by British authors published between 1900 and 1950 not set in England</query>
<passage>Grand Slam tennis tournaments — the Australian Open, French Open, Wimbledon, and US Open — are held annually and comprise the four most prestigious events in professional tennis.</passage>
<grade>0</grade>
<justification>The passage discusses tennis — no connection to British novels.</justification>
</example>

<example>
<query>novels by British authors published between 1900 and 1950 not set in England</query>
<passage>British literature in the first half of the twentieth century saw major shifts in style and subject, with modernist authors like Virginia Woolf and James Joyce reshaping narrative form alongside Bloomsbury Group writers and later post-war voices.</passage>
<grade>1</grade>
<justification>The passage is about British literature in the target era but applies no set-filter (British + 1900-1950 + non-English setting) — related, does not answer.</justification>
</example>

<example>
<query>novels by British authors published between 1900 and 1950 not set in England</query>
<passage>Several British novels from the 1900-1950 period are set outside England. Joseph Conrad's Lord Jim (1900) takes place in Indonesia; Somerset Maugham's The Painted Veil (1925) is set in China; George Orwell's Burmese Days (1934) is set in Burma. British literature more broadly during this period was dominated by modernism, with Woolf and Joyce redefining form, while Edwardian-era novelists like Kipling and Wells wrote extensively on empire and technology respectively.</passage>
<grade>2</grade>
<justification>The passage answers with a partial list matching all filters, but also wanders into modernism and Edwardian-era commentary unrelated to the set query — extra information beyond the direct answer.</justification>
</example>

<example>
<query>novels by British authors published between 1900 and 1950 not set in England</query>
<passage>British novels published 1900-1950 with non-English settings include: Joseph Conrad's Lord Jim (1900, Indonesia); Somerset Maugham's The Moon and Sixpence (1919, Tahiti) and The Painted Veil (1925, China); Rudyard Kipling's Kim (1901, India); George Orwell's Burmese Days (1934, Burma) and Down and Out in Paris and London (1933, France); Graham Greene's The Power and the Glory (1940, Mexico).</passage>
<grade>3</grade>
<justification>The passage is dedicated to the set query — every entity satisfies all filters, no tangential content.</justification>
</example>
</examples>

Notice: the same information can appear at different grades depending on focus. Grade 2 and grade 3 both contain the answer, but grade 2 embeds it among tangentially related content, while grade 3 stays focused on what the query asks. **When a passage answers the query but also discusses related-but-not-asked material, use grade 2. Do not collapse grade 2 into grade 1 or grade 3.**

Now assess the following pair:"""

PROMPT_FEW_SHOT = PROMPT + _FEW_SHOT_EXAMPLES


def build_fewshot_prompt(
    examples: list[tuple[str, str, int, str]],
) -> str:
    """Build a few-shot judge prompt from (query, passage, grade,
    justification) tuples. Prepended to the base TREC-DL rubric.

    Use for the real-exemplar ablation path deferred by d36: sample
    real TREC-DL judgments with a different seed than the calibration
    sample (so the example pairs are disjoint), fetch their doc texts,
    pass them here to get a domain-anchored prompt.
    """
    example_blocks = "\n\n".join(
        f"<example>\n<query>{query}</query>\n<passage>{passage}</passage>\n"
        f"<grade>{grade}</grade>\n<justification>{justification}</justification>\n</example>"
        for query, passage, grade, justification in examples
    )
    return (
        PROMPT
        + "\n\n<examples>\n"
        + example_blocks
        + "\n</examples>\n\nNow assess the following pair:"
    )


def judge(
    query: str,
    doc: str,
    model_id: str,
    *,
    client: Anthropic | None = None,
    prompt: str = PROMPT,
) -> LLMJudgment:
    """Grade one (query, doc) pair with the given prompt.

    Defaults to the base TREC-DL guidelines (d35). Swap to
    `PROMPT_FEW_SHOT` (d36) when the base prompt shows middle-grade
    avoidance, or to a custom prompt from `build_fewshot_prompt()` for
    the real-exemplar ablation.

    `client` is optional so a caller can share one `Anthropic()`
    instance across many calls (recommended for the pilot's batch).
    """
    client = client or Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        temperature=0.0,
        system=[{
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "record_relevance"},
        messages=[{
            "role": "user",
            "content": f"Query: {query}\n\nPassage: {doc}",
        }],
    )
    tool_use = next(
        block for block in response.content if block.type == "tool_use"
    )
    return LLMJudgment.model_validate(tool_use.input)
